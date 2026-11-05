//! Elasticsearch 8.x-compatible REST server.
//!
//! One axum fallback handler feeds every request into [`dispatch`], a
//! hand-written router over path segments (ES's path grammar has too many
//! overlapping shapes for a declarative router). Handlers return [`Resp`];
//! [`finish`] applies the cross-cutting HTTP behaviour: `X-Elastic-Product`,
//! vendor media types, `?pretty`, `?filter_path`, `?error_trace`, HEAD,
//! gzip.

mod async_search;
pub mod mget_body;
mod input_metadata;
mod error_admitted;
mod response_tree;
mod cat;
pub mod compression;
/// Applied cluster metadata for cat aliases, independent of local shard recovery.
/// Only an in-process coordinator can attach this catalog.
#[derive(Clone)]
pub struct AliasCatalog(pub Vec<(String, bool, std::collections::BTreeMap<String, serde_json::Value>)>);
/// Names that exist in applied cluster metadata, even during local recovery.
/// Only trusted in-process frontends can install this authorization context.
#[derive(Clone)]
pub struct AuthorizationIndexCatalog(pub std::collections::BTreeSet<String>);
mod usage;
mod cat_extra;
mod cluster;
mod ccs;
mod deprecations;
mod coordination;
mod desired_nodes;
mod features;

/// The single-node capability table (`GET /_capabilities`): `supported` for
/// `method` + `path` with `params` and `caps`, `None` for an unknown route.
pub fn capability_supported(method: &str, path: &str, params: &[&str], caps: &[&str]) -> Option<serde_json::Value> {
    features::capability_supported(method, path, params, caps)
}

/// Features of the engine served by this binary, for cluster-state discovery.
pub fn node_features(node_id: &str) -> serde_json::Value {
    features::nodes_features(node_id, None)[0].clone()
}

pub fn local_node_features(node: &elasticsearch_engine::node::Node) -> Vec<&'static str> {
    features::local_features(node)
}

/// Coordinator validation uses the same installed query parsers as engine searches.
/// The thread-local module scope ends before the caller can await or spawn work.
pub fn validate_search_request(node: &elasticsearch_engine::node::Node, body: &serde_json::Value,
    params: &std::collections::HashMap<String, String>) -> Result<elasticsearch_engine::search::SearchRequest, elasticsearch_engine::error::EsError> {
    let _modules = search_api::test_modules(node);
    let request = elasticsearch_engine::search::parse_request(body, params)?;
    search_api::search_validate::validate_request(body, params, &request)?;
    Ok(request)
}

/// Shared validation before the cluster publishes profiling resource creation.
pub fn validate_profiling_status(path: &str, params: &std::collections::HashMap<String, String>) -> Result<(), elasticsearch_engine::error::EsError> {
    profiling::status_options(path, params).map(|_| ())
}

mod datastream;
mod docs;
mod indices;
mod ingest;

/// Ingest already executed by a trusted cluster coordinator, by bulk position.
/// Network clients cannot attach this extension.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum PreparedBulkIngest {
    Write { routing: Option<String>, dynamic_templates: Option<std::collections::HashMap<String, String>> },
    Dropped,
    Failed(elasticsearch_engine::ingest::IngestFailure),
}
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PreparedBulkIngestItems(pub std::collections::BTreeMap<usize, PreparedBulkIngest>);

pub fn ingest_dropped_response(index: &str, id: Option<&str>) -> Value {
    Value::Object(ingest::dropped_json(index, id))
}

mod kibana_compat;
mod analytics;
mod connector;
mod fleet;
mod misc9;
mod monitoring;
mod profiling;
mod inference;
mod ml;
mod mvt;
mod obs_search;
mod observability;
pub mod pools;
mod query_lang;
mod request_cache;
mod scripts;
mod search_api;
pub mod search_task;
mod snapshot_api;
pub mod security;
/// Trusted native-user API projection into the real security-index primary.
#[derive(Clone)]
pub struct NativeSecurityMirror(pub Arc<dyn Fn(&str, Option<Value>) -> Result<(), EsError> + Send + Sync>);
/// The node's enrollment material for `GET /_security/enroll/{node,kibana}`
/// (ES `TransportNodeEnrollmentAction` / `TransportKibanaEnrollmentAction`),
/// attached by the cluster layer to `/_security/enroll/*` requests: whether
/// `xpack.security.enrollment.enabled` is set, the node enrollment response
/// body, and the base64 (DER) HTTP CA certificate for Kibana. Without it the
/// APIs answer as with enrollment disabled.
#[derive(Clone)]
pub struct Enrollment {
    pub enabled: bool,
    pub node: Arc<dyn Fn() -> Result<Value, EsError> + Send + Sync>,
    pub http_ca: Arc<dyn Fn() -> Result<String, EsError> + Send + Sync>,
}
/// Failure documents are acknowledged by their own destination primary.
#[derive(Clone)]
pub struct FailureStoreWriter(pub Arc<dyn Fn(&str, Value, Option<usize>) -> Result<Value, EsError> + Send + Sync>);


mod spec_routes;
mod spec_validate;

/// ES `BaseRestHandler`: the 400 a request with a body gets on a route that
/// takes none (`request [M /p] does not support having a body`), for a
/// coordinator to answer before fanning the request out to shards (whose
/// own rejection would otherwise be a shard failure inside a 200).
pub fn body_not_supported(method: &axum::http::Method, path: &str, version: elasticsearch_engine::version::RestApiVersion) -> Option<EsError> {
    spec_validate::body_rejection(method, path, version)
}
mod tasks;
pub mod tls;
pub use tls::TlsListener;
mod termvectors;
mod sysinfo;
pub use sysinfo::{configure_processors, set_memory_locked};
mod util;
pub mod typed_bulk;
mod enrich;
mod ccr;
mod graph;
mod rollup;
mod autoscaling;
mod ilm;
mod transform;
pub mod watcher;
mod text_structure;
mod logstash;
mod license;
mod system_features;
pub use system_features::feature_system_indices;
pub mod outgoing_headers;
pub mod wire;
pub mod xcontent_in;

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Method, Request, Response, StatusCode};
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::Node;
pub use elasticsearch_engine::version::RestApiVersion;
use serde_json::Value;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

pub use util::{filter_path, humanize};
pub use async_search::untype_keys as untype_search_keys;
pub use search_api::{apply_search_routing, rank_license as check_search_license, MsearchBreaker};

/// Trusted in-process lookup for a connector stored on another primary.
#[derive(Clone)]
pub struct ConnectorSourceLookup(pub Arc<dyn Fn(&str) -> Result<Option<Value>, EsError> + Send + Sync>);

/// Trusted coordinator execution after application template and workflow validation.
#[derive(Clone)]
pub struct SearchApplicationExecutor(pub Arc<dyn Fn(&[String], &Value, &HashMap<String, String>) -> Result<String, EsError> + Send + Sync>);
/// Trusted coordinator search over every shard of the targets (index
/// expression, body), for APIs that run searches internally (`_rank_eval`).
#[derive(Clone)]
pub struct ClusterSearchExecutor(pub Arc<dyn Fn(Option<&str>, &Value) -> Result<String, EsError> + Send + Sync>);
#[derive(Clone)]
pub struct PreparedSearchApplication {
    pub indices: Vec<String>,
    pub query: Value,
}

#[derive(Clone)]
struct DecodedSourceOwner(std::sync::Arc<elasticsearch_engine::memory::Reservation>);

/// A parsed request.
pub struct Req {
    pub method: Method,
    pub path: String,
    pub segs: Vec<String>,
    pub params: HashMap<String, String>,
    pub body: Bytes,
    pub headers: HeaderMap,
    /// Security: what the caller may read (set by the authorization layer for
    /// every non-superuser request; `None` with security off or for superusers).
    pub access: Option<Arc<security::access::ReadAccess>>,
    /// The http request's extensions: in-process callers (the cluster
    /// coordinator's `router.oneshot`) attach [`DfsStats`] / [`DfsPhaseOnly`]
    /// here. Unreachable from the network.
    pub extensions: axum::http::Extensions,
    /// The REST API version the request asked for (`compatible-with=8` on
    /// `Accept` / `Content-Type` selects [`RestApiVersion::V8`]; ES's
    /// `RestRequest.getRestApiVersion()`).
    pub api_version: RestApiVersion,
}

/// DFS statistics of `dfs_query_then_fetch` (set as an http request
/// extension `Arc<DfsStats>` on `_search` to score with them).
pub use elasticsearch_engine::search::phases::{merge_dfs, DfsField, DfsStats, DfsTerm};

/// Http request extension: `_search` runs only its DFS phase and answers
/// `200` with this node's [`DfsStats`] (serde JSON) for the request's
/// shards, after the full request validation, resolution and security.
#[derive(Clone, Copy, Debug, Default)]
pub struct DfsPhaseOnly;

/// Http request extension (in-process callers only): a plain JSON response
/// (no `pretty` / `human` / `filter_path` / other format, not `HEAD`, no
/// gzip) is handed back unserialized, as a [`JsonResponse`] response
/// extension with an empty body. Callers that parse the response anyway
/// (the cluster coordinator merging per-shard `_bulk` responses) skip a
/// serialize + parse round trip.
#[derive(Clone, Copy, Debug, Default)]
pub struct InProcessJson;

/// Http request extension on an in-process `_bulk` (the cluster
/// coordinator's own shard group): item `i` (in body order) carries an
/// `_id` the coordinator generated for this request, on its first attempt,
/// so no earlier copy of the document can exist and the shard skips the
/// version lookup (ES's append-only optimisation for auto-generated ids).
#[derive(Clone, Debug)]
pub struct AppendOnlyItems(pub std::sync::Arc<Vec<bool>>);

/// Http request extension on an in-process `_bulk` (the cluster
/// coordinator's own shard group): the items the coordinator
/// already split and validated, in body order, so the engine does not parse
/// the NDJSON a second time. The request body is then empty; `bytes` is the
/// size it would have had (indexing pressure). The handler takes the items
/// once; a request that never reached it (or a retry after it did) is sent
/// with its body.
#[derive(Clone)]
pub struct PreparedBulk {
    items: Arc<std::sync::Mutex<Option<Vec<PreparedBulkItem>>>>,
    pub bytes: u64,
    /// Where the handler leaves typed item results ([`typed_bulk`]) when
    /// the coordinator asked for them ([`PreparedBulk::typed`]).
    typed: Option<Arc<std::sync::Mutex<Option<typed_bulk::TypedBulk>>>>,
}

/// One coordinator-parsed bulk item: the action line's operation and
/// parameters, and the source line (`None` for a delete).
#[derive(Clone, Debug)]
pub struct PreparedBulkItem {
    pub action: String,
    pub meta: serde_json::Map<String, Value>,
    pub source: Option<String>,
}

impl PreparedBulk {
    pub fn new(items: Vec<PreparedBulkItem>, bytes: u64) -> Self {
        PreparedBulk { items: Arc::new(std::sync::Mutex::new(Some(items))), bytes, typed: None }
    }

    /// Ask the handler for typed item results ([`typed_bulk`]), collected
    /// with [`PreparedBulk::take_typed`].
    pub fn typed(mut self) -> Self {
        self.typed = Some(Arc::new(std::sync::Mutex::new(None)));
        self
    }

    /// Whether the coordinator takes typed results.
    pub fn wants_typed(&self) -> bool {
        self.typed.is_some()
    }

    /// Leave the typed results (the handler, instead of rendering them).
    pub fn put_typed(&self, typed: typed_bulk::TypedBulk) -> Result<(), typed_bulk::TypedBulk> {
        match &self.typed {
            Some(slot) => match slot.lock() { Ok(mut slot) => { *slot = Some(typed); Ok(()) } Err(_) => Err(typed) },
            None => Err(typed),
        }
    }

    /// The typed results the handler left, if it did.
    pub fn take_typed(&self) -> Option<typed_bulk::TypedBulk> {
        self.typed.as_ref()?.lock().ok()?.take()
    }

    /// The items are still there (no handler took them).
    pub fn is_pending(&self) -> bool {
        self.items.lock().map(|items| items.is_some()).unwrap_or(false)
    }

    /// Take the items (once).
    pub fn take(&self) -> Option<Vec<PreparedBulkItem>> {
        self.items.lock().ok()?.take()
    }
}

/// Request header of a cluster coordinator's composite scroll asking a
/// page's hit spans, and the response header carrying them
/// ([`encode_hit_spans`]): internal, stripped from client requests.
pub const HIT_SPANS_HEADER: &str = "x-rustic-internal-hit-spans";

/// `tail,len,len,...`: the bytes after the last hit of a response's
/// `hits.hits` to the end of the body, then each hit's length (hits are
/// separated by one comma). Counted from the end, the spans survive edits
/// of the body before the hits (a scroll id given its node prefix).
pub fn encode_hit_spans(tail: usize, lens: &[u32]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(8 + lens.len() * 4);
    let _ = write!(out, "{tail}");
    for len in lens {
        let _ = write!(out, ",{len}");
    }
    out
}

/// Response extension carrying the JSON body, see [`InProcessJson`].
#[derive(Clone, Debug)]
pub struct JsonResponse(pub Value);

/// Http request extension (in-process callers only, with
/// [`handle_in_process`]): a plain response (as for [`InProcessJson`])
/// whose handler rendered its body as text (a search's) is handed back as a
/// [`TypedResponse`] extension of an empty response, instead of as an HTTP
/// response whose header map and body the caller would take apart again.
#[derive(Clone, Copy, Debug, Default)]
pub struct InProcessTyped;

/// Response extension of an [`InProcessTyped`] request: the status, the
/// headers as a caller reads them off the HTTP response (lowercase names,
/// one name's values together in the order of its first, values that are
/// not visible ASCII left out) and the body.
#[derive(Clone, Debug)]
pub struct TypedResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

/// The headers of an HTTP response built from `headers`, as
/// [`TypedResponse`] holds them; `None` when building that response would
/// fail (a name or value HTTP does not allow).
fn typed_headers<'a>(headers: impl Iterator<Item = (&'a str, &'a str)> + Clone) -> Option<Vec<(String, String)>> {
    fn tchar(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)
    }
    let mut out: Vec<(String, String)> = Vec::new();
    for (name, value) in headers.clone() {
        if name.is_empty() || !name.bytes().all(tchar) || !value.bytes().all(|b| (b >= 32 && b != 127) || b == b'\t') {
            return None;
        }
    }
    // A header map keeps one entry per name, its values in order, entries
    // in the order of each name's first value.
    let mut names: Vec<String> = Vec::new();
    for (name, _) in headers.clone() {
        let lower = name.to_ascii_lowercase();
        if !names.contains(&lower) {
            names.push(lower);
        }
    }
    for name in names {
        for (_, value) in headers.clone().filter(|(n, _)| n.eq_ignore_ascii_case(&name)) {
            // `HeaderValue::to_str`: visible ASCII only.
            if value.bytes().all(|b| b == b'\t' || (32..127).contains(&b)) {
                out.push((name.clone(), value.to_string()));
            }
        }
    }
    Some(out)
}

pub enum RespBody {
    Json(Value),
    /// Parsed coordinator output retaining its source/header allocation owner.
    RetainedJson { value: Value, owner: Bytes },
    /// Pre-serialized JSON (search responses with spliced `_source`).
    Raw(String),
    Text(String),
    /// Bytes with a handler-set `content-type` header (csv, yaml, cbor, ...).
    Bytes(Vec<u8>),
    /// Already admitted coordinator bytes; the Bytes owner survives streaming.
    /// JSON retains the same pretty/filter/wire handling as Raw.
    Owned { bytes: Bytes, json: bool },
    Empty,
}

impl RespBody {
    pub(crate) fn retained_json(value: Value, owner: Option<Bytes>) -> Self {
        match owner { Some(owner) => Self::RetainedJson { value, owner }, None => Self::Json(value) }
    }
}

pub struct Resp {
    pub status: u16,
    pub headers: Vec<(&'static str, String)>,
    // Headers are destroyed before a body that may own their admission.
    pub body: RespBody,
}

impl Resp {
    pub fn json(status: u16, v: Value) -> Resp {
        Resp { status, body: RespBody::Json(v), headers: Vec::new() }
    }
    pub fn ok(v: Value) -> Resp {
        Resp::json(200, v)
    }
    pub fn raw(status: u16, s: String) -> Resp {
        Resp { status, body: RespBody::Raw(s), headers: Vec::new() }
    }
    pub fn text(s: String) -> Resp {
        Resp { status: 200, body: RespBody::Text(s), headers: Vec::new() }
    }
    pub fn empty(status: u16) -> Resp {
        Resp { status, body: RespBody::Empty, headers: Vec::new() }
    }
    pub fn ack() -> Resp {
        Resp::ok(serde_json::json!({"acknowledged": true}))
    }
}

pub type R = Result<Resp, EsError>;

/// An in-process continuation after request validation and authorization.
/// Network clients cannot install extensions; the cluster frontend uses this
/// to apply the same security policy before its own coordinator executes.
#[derive(Clone)]
pub struct AuthorizedDispatch(pub Arc<dyn Fn(Req) -> std::pin::Pin<Box<dyn std::future::Future<Output = R> + Send>> + Send + Sync>);

#[derive(Clone, Copy)]
pub struct ReadShardSelection(pub u32, pub Option<usize>);
tokio::task_local! { static READ_SHARD: Option<(u32, Option<usize>)>; }
tokio::task_local! { static ENGINE_POOL: pools::Pool; }
tokio::task_local! { static NESTED: bool; }

/// Http request extension: this request is a sub-request whose result some
/// executor thread (on this node or another) waits for. Its engine work
/// runs off the executors ([`pools::run_unpooled`]).
#[derive(Clone, Copy, Debug)]
pub struct NestedRequest;

/// Http request extension: a cluster coordinator's shard-level write
/// executed on the node that coordinates it (its coordinating stage holds
/// the bytes already: the primary stage counts them once).
#[derive(Clone, Copy, Debug)]
pub struct CoordinatorLocal;

/// Trusted coordinator expansion of query-language source targets. The
/// query handler applies the caller's DLS/FLS after expansion.
#[derive(Clone)]
pub struct QueryTargetExpander(pub Arc<dyn Fn(Vec<elasticsearch_engine::search::Target>) -> Result<Vec<elasticsearch_engine::search::Target>, EsError> + Send + Sync>);

/// In-process node telemetry gathering after frontend authorization. It is
/// restricted to GET /_nodes/_local and cannot be supplied by network clients.
#[derive(Clone, Copy)]
pub struct InternalNodeMonitor;

/// Internal REST execution is transport work, not a new HTTP connection.
#[derive(Clone, Copy)]
pub struct InternalHttpDispatch;
pub use observability::{HttpConnection, http_request};

/// Cluster-internal work running as ES's `_system` user (no caller
/// credentials). Only the cluster layer attaches it, for requests that
/// arrived over the node-to-node transport carrying its internal marker;
/// network clients cannot supply extensions.
#[derive(Clone, Copy)]
pub struct SystemIdentity;

/// Runs a REST request through the cluster coordinator as the system
/// (after authorization): `(method, uri, body)` → response body, or the
/// response's error. Engine work that creates or fills indices on a cluster
/// (enrich policy execution, downsampling) goes through it so it reaches
/// every shard. A `Value::String` body is sent verbatim as NDJSON (`_bulk`).
#[derive(Clone)]
pub struct ClusterRest(pub Arc<dyn Fn(&str, &str, Option<Value>) -> Result<Value, EsError> + Send + Sync>);

/// Several [`ClusterRest`] calls `(method, uri, body)` run at once (their
/// answers in order): a retriever's plan phase runs its leaf children's
/// searches concurrently, as ES runs a compound retriever's child searches
/// in one multi-search. Only attached to a rank plan phase.
#[derive(Clone)]
pub struct ClusterRestMany(pub Arc<dyn Fn(Vec<(String, String, Option<Value>)>) -> Vec<Result<Value, EsError>> + Send + Sync>);

/// A coordinated response retained as `(HTTP status, JSON text)`, including errors.
/// Search responses may contain
/// lone UTF-16 units and original number spellings that `Value` cannot retain.
#[derive(Clone)]
pub struct ClusterRestRaw(pub Arc<dyn Fn(&str, &str, Option<Value>) -> Result<(u16, String), EsError> + Send + Sync>);


/// [`ClusterRest`] running as the internal `_system` user, for work ES runs
/// under an internal origin once the triggering request was authorized
/// (enrich policy execution, downsampling): the caller needs no privileges
/// on the indices that work creates or fills.
#[derive(Clone)]
pub struct InternalClusterRest(pub Arc<dyn Fn(&str, &str, Option<Value>) -> Result<Value, EsError> + Send + Sync>);

impl Req {
    /// The internal-origin coordinator call when the cluster layer provides
    /// one, else the caller's.
    pub(crate) fn internal_cluster_rest(&self) -> Option<ClusterRest> {
        self.extensions.get::<InternalClusterRest>().map(|r| ClusterRest(r.0.clone())).or_else(|| self.extensions.get::<ClusterRest>().cloned())
    }
}

/// Whether the cluster task this request runs for has been cancelled (only
/// the cluster front end attaches this, to copy-local `_msearch` batches):
/// the engine checks it before each item.
#[derive(Clone)]
pub struct ParentCancelled(pub Arc<dyn Fn() -> bool + Send + Sync>);

/// The cluster task a search shard request runs for (only the cluster front
/// end attaches this): the engine registers the shard-level child task with
/// this `parent_task_id`, and its cancellation follows `token` (the
/// parent's, when the parent runs in this process; a cancelled one when
/// the parent was banned before the request arrived). See
/// [`search_task`](crate::search_task).
#[derive(Clone)]
pub struct SearchParent {
    pub id: String,
    pub token: Option<Arc<elasticsearch_engine::rustic_core::cancel::Token>>,
}

/// Search usage (`xpack.usage` analytics) is counted by the cluster
/// coordinator that received the client request, not by each shard copy.
#[derive(Clone, Copy)]
pub struct CoordinatorUsage;

/// A cluster coordinator's phases of a `retriever` / `rank` search over
/// several shards (only the cluster front end attaches this).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InternalRankPlan {
    /// Run the children (through [`ClusterRest`]) and answer
    /// `{"_rustic_final_body": <body of the final search>}`.
    Plan,
    /// The body may carry a serialized rank plan
    /// (`elasticsearch_engine::retriever::RANK_PLAN_KEY`).
    Apply,
}

/// A cluster coordinator renewing a scroll sub-context (only the cluster
/// front end attaches this): `_search/scroll` renews the context's expiry
/// and keep-alive and answers without reading a page.
#[derive(Clone, Copy, Debug)]
pub struct InternalScrollTouch;

/// Tests only: answer a scroll touch with a page, as a node of a build
/// before the touch does (a rolling upgrade).
#[doc(hidden)]
pub static SCROLL_TOUCH_AS_PAGE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A cluster coordinator's query_then_fetch phases of a shard search (only
/// the cluster front end attaches this).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InternalQtf {
    /// Search a pinned view, register it as a reader context and answer
    /// with `"_rustic_ctx"` and each hit's `"_rustic_doc"`.
    Query,
    /// The body carries [`QTF_FETCH_KEY`] `{"ctx": id, "docs": [[target,
    /// leaf, doc, score], ...]}`: render those hits of that context and
    /// free it (no docs: only free it).
    Fetch,
    /// A shard whose query phase kept no context, searched again in one
    /// phase (no aggregations) to render its winners: counted as its fetch
    /// phase.
    Rerun,
    /// A small page answered in one round (ES `query_and_fetch`): search
    /// and render the shard's top `from + size` hits as a plain search
    /// does, but count only the query phase and answer with
    /// [`QTF_FETCH_NS_KEY`] first, the time spent rendering. The
    /// coordinator counts the fetch on the shards whose hits made the
    /// merged page, as ES counts a query_then_fetch fetch phase.
    QueryFetch,
}

/// First member of a [`InternalQtf::QueryFetch`] answer: the nanoseconds
/// the shard spent rendering its hits (its fetch time, counted by the
/// coordinator when the shard's hits make the page).
pub const QTF_FETCH_NS_KEY: &str = "_rustic_fetch_ns";

/// The reader context id a query_then_fetch query phase registers, chosen
/// by the coordinator (its own internal header beside the phase's): a copy
/// it gave up on can then be told to free it, before or after the phase
/// arrives.
#[derive(Clone, Debug)]
pub struct InternalQtfCtx(pub String);

/// Set on a query_then_fetch query phase from a coordinator that runs
/// pinned query_then_fetch (`QtfPinnedV1`): it never re-runs a shard, so a
/// reader context the shard cannot admit fails the shard. An older
/// coordinator re-runs a shard answering without a context in one phase,
/// and gets that answer.
#[derive(Clone, Copy, Debug)]
pub struct InternalQtfPinned;

/// Http request extension of a query_then_fetch phase from a coordinator
/// that merges light query hits and matches fetched hits by address
/// ([`elasticsearch_engine::search::SearchRequest::light_refs`] and
/// `fetch_refs`).
#[derive(Clone, Copy, Debug)]
pub struct InternalQtfLight;

/// Body key of a query_then_fetch fetch phase (see [`InternalQtf::Fetch`]).
pub const QTF_FETCH_KEY: &str = "_rustic_fetch";

pub use usage::{analytics_counts, record_analytics, xpack_usage};
pub use observability::{nodes_stats_in_process, search_usage_json};
pub use indices::{index_stats_in_process, vector_counts_in};

/// A coordinator-routed write to a data-stream backing index. Only the
/// cluster front end may attach this extension after validating the original
/// data-stream request.
#[derive(Clone, Copy)]
pub struct InternalDataStreamBackingWrite;

/// In-process extension (cluster front end only): the request body as the
/// front end already parsed it with [`jsonpos::parse_plain`] (plain JSON, no
/// duplicate keys), which is exactly what [`parse_body`] returns for it.
/// [`Req::json_shared`] shares it (every shard request of a search reads the
/// coordinator's one parse; a kNN body is thousands of floats, and copying
/// it per shard cost ~3% of a hybrid search node's CPU), [`Req::json`]
/// copies it, if it came with this very body; both parse as before
/// otherwise. Unreachable from the network.
#[derive(Clone)]
pub struct PreparsedBody(Arc<(Bytes, Arc<Value>)>);
impl PreparsedBody {
    pub fn new(body: Bytes, value: Arc<Value>) -> Self { Self(Arc::new((body, value))) }
    fn shared_for(&self, body: &Bytes) -> Option<Arc<Value>> {
        // The same buffer (a shard request's body is the coordinator's), or
        // the same bytes.
        if self.0.0.as_ptr() != body.as_ptr() || self.0.0.len() != body.len() {
            if self.0.0.as_ref() != body.as_ref() { return None; }
        }
        Some(self.0.1.clone())
    }
}

impl Req {
    /// Body as JSON (`Null` when empty); `?source=` is the GET-body fallback.
    pub fn json(&self) -> Result<Value, EsError> {
        if self.body.is_empty() {
            if let Some(s) = self.params.get("source") {
                return parse_body(s.as_bytes());
            }
            return Ok(Value::Null);
        }
        if self.body.iter().all(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r')) {
            return Ok(Value::Null);
        }
        if let Some(value) = self.extensions.get::<PreparsedBody>().and_then(|p| p.shared_for(&self.body)) {
            return Ok(Arc::try_unwrap(value).unwrap_or_else(|shared| (*shared).clone()));
        }
        parse_body(&self.body)
    }

    /// [`Req::json`], shared: the coordinator's parse ([`PreparsedBody`])
    /// without a copy.
    pub fn json_shared(&self) -> Result<Arc<Value>, EsError> {
        if !self.body.is_empty() {
            if let Some(value) = self.extensions.get::<PreparsedBody>().and_then(|p| p.shared_for(&self.body)) {
                return Ok(value);
            }
        }
        self.json().map(Arc::new)
    }

    /// The text request-body positions are resolved against (`?source=` for GET bodies).
    pub fn body_text(&self) -> &[u8] {
        if self.body.is_empty() {
            if let Some(s) = self.params.get("source") {
                return s.as_bytes();
            }
        }
        &self.body
    }

    pub fn body_str(&self) -> Result<&str, EsError> {
        std::str::from_utf8(&self.body).map_err(|_| EsError::parsing("request body is not valid UTF-8"))
    }

    pub fn param(&self, k: &str) -> Option<&str> {
        self.params.get(k).map(String::as_str)
    }

    /// Boolean query parameter: only `true`, `false` or empty (= true).
    pub fn bool_param(&self, k: &str) -> Result<Option<bool>, EsError> {
        match self.params.get(k).map(String::as_str) {
            None => Ok(None),
            Some("true") | Some("") => Ok(Some(true)),
            Some("false") => Ok(Some(false)),
            Some(v) => Err(EsError::illegal_argument(format!(
                "Failed to parse value [{v}] as only [true] or [false] are allowed."
            ))),
        }
    }

    pub fn flag(&self, k: &str) -> Result<bool, EsError> {
        Ok(self.bool_param(k)?.unwrap_or(false))
    }

    pub fn seg(&self, i: usize) -> &str {
        self.segs.get(i).map(String::as_str).unwrap_or("")
    }

    /// True when the client asked for the 8.x REST contract
    /// (`compatible-with=8`): the 8.x behaviour where 9.x changed it.
    pub fn compat_v8(&self) -> bool {
        self.api_version.is_v8()
    }
}

/// Parse a JSON body exactly as ES's Jackson-backed parser would fail:
/// syntax errors carry Jackson's message and location, duplicate keys and
/// trailing tokens are rejected.
pub fn parse_body(text: &[u8]) -> Result<Value, EsError> {
    use elasticsearch_engine::jsonpos::{self, Trailing};
    // Plain JSON parses in one pass; anything else takes the Jackson path.
    if let Some(v) = jsonpos::parse_plain(text).filter(|v| !v.is_null()) {
        return Ok(v);
    }
    match jsonpos::validate(text) {
        Err(e) => return Err(e.to_error(text)),
        Ok(Trailing::Token(kind, pos)) => {
            return Err(EsError::parsing(format!("Unexpected token [{kind}] found after the main object.")).with("line", pos.line).with("col", pos.col));
        }
        Ok(Trailing::None) => {}
    }
    let stripped = jsonpos::strip_comments(text);
    let text = stripped.as_deref().unwrap_or(text);
    let s = std::str::from_utf8(text).map_err(|_| EsError::parsing("request body is not valid UTF-8"))?;
    let v: Value = serde_json::from_str(s).map_err(|e| EsError::json_parse(&e))?;
    if v.is_null() {
        // A literal `null` body is not "no body" (ES: `Expected [START_OBJECT] but found [VALUE_NULL]`).
        return Err(EsError::parsing("Expected [START_OBJECT] but found [VALUE_NULL]").cursor("root").with_lc());
    }
    Ok(v)
}

/// Run blocking engine work off the async runtime.
/// Records an executor job's queue wait and run time (benchmarks).
struct HopTimer(u128, std::time::Instant);
impl Drop for HopTimer {
    fn drop(&mut self) {
        elasticsearch_engine::search::search_phases::record_named("engine_executor", vec![("queue", self.0), ("run", self.1.elapsed().as_micros())]);
    }
}

pub async fn blocking<F>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
{
    let selected = READ_SHARD.try_with(|s| *s).ok().flatten();
    let inf = elasticsearch_engine::inference::scoped_current();
    let coordinator = elasticsearch_engine::coord::current();
    let api_version = elasticsearch_engine::version::current_rest_api_version();
    let memory = elasticsearch_engine::memory::current_scope();
    // The executor the request's route selected (see `pools::pool_for`);
    // work outside a request (background tasks) runs on `management`.
    let pool = ENGINE_POOL.try_with(|p| *p).unwrap_or(pools::Pool::Management);
    let submitted = std::time::Instant::now();
    let job = elasticsearch_engine::memory::inherit_request_policy(move || {
        let started = submitted.elapsed().as_micros();
        let _hop = HopTimer(started, std::time::Instant::now());
        let _shard = elasticsearch_engine::search::ReadShardScope::enter(selected.map(|s| s.0));
        let _count = elasticsearch_engine::search::ReadShardCountScope::enter(selected.and_then(|s| s.1));
        elasticsearch_engine::memory::scoped(memory, || {
            util::collect_deferred_refreshes(|| elasticsearch_engine::coord::scoped(coordinator, || {
                elasticsearch_engine::version::with_rest_api_version(api_version, || elasticsearch_engine::inference::scoped(inf, f))
            }))
        })
    });
    let blocking_started = std::time::Instant::now();
    let ran = if NESTED.try_with(|n| *n).unwrap_or(false) {
        // A nested write (a reindex / by-query / fan-out sub-bulk) runs off
        // the write executor but is still rejected while its queue is full.
        let _admitted = if pool == pools::Pool::Write { Some(pools::executor(pool).admit()?) } else { None };
        pools::run_unpooled(job).await
    } else { pools::run(pool, job).await };
    elasticsearch_engine::search::search_phases::record_named("engine_blocking", vec![("submit_to_return", blocking_started.elapsed().as_micros())]);
    let (r, pending) = match ran {
        Ok(out) => out,
        Err(e) => return Err(e),
    };
    // `refresh=wait_for`: the executor thread is free; respond once the
    // refreshes that make the writes visible completed (ES RefreshListeners).
    let mut forced = false;
    for rx in pending {
        match rx.await {
            Ok(Ok(f)) => forced |= f,
            Ok(Err(error)) => return Err(error),
            Err(_) => return Err(EsError::internal("refresh listener dropped")),
        }
    }
    let mut r = r;
    if forced {
        if let Ok(resp) = &mut r {
            mark_forced_refresh(resp);
        }
    }
    r
}

/// Set `forced_refresh` on a write response whose wait_for refresh had to
/// be forced: on the response, or on each successful bulk item.
fn mark_forced_refresh(resp: &mut Resp) {
    let RespBody::Json(Value::Object(body)) = &mut resp.body else { return };
    if let Some(items) = body.get_mut("items").and_then(Value::as_array_mut) {
        for item in items {
            if let Some(op) = item.as_object_mut().and_then(|o| o.values_mut().next()).and_then(Value::as_object_mut) {
                if op.get("error").is_none() { op.insert("forced_refresh".into(), Value::Bool(true)); }
            }
        }
    } else {
        body.insert("forced_refresh".into(), Value::Bool(true));
    }
}

fn no_handler(req: &Req) -> Resp {
    Resp::json(
        400,
        serde_json::json!({
            "error": format!("no handler found for uri [{}] and method [{}]", req.path, req.method)
        }),
    )
}

async fn dispatch(node: Arc<Node>, mut req: Req) -> R {
    let selected = req.extensions.remove::<ReadShardSelection>().map(|s| (s.0, s.1));
    // A retriever's plan phase coordinates child searches and waits for
    // them: on `search_coordination`, never on a `search` thread the
    // children's shard phases need (see `pools::Pool::SearchCoordination`).
    let pool = if req.extensions.get::<InternalRankPlan>() == Some(&InternalRankPlan::Plan) { pools::Pool::SearchCoordination } else { pools::pool_for(&req.method, &req.segs) };
    // Byte-based indexing pressure (ES `indexing_pressure.memory.limit`):
    // document writes hold their request bytes while they run.
    let _pressure = if pool == pools::Pool::Write {
        elasticsearch_engine::indexing_pressure::configure_from(&node);
        let bytes = req.body.len() as u64 + req.extensions.get::<PreparedBulk>().map_or(0, |p| p.bytes);
        let pressure = elasticsearch_engine::indexing_pressure::global();
        if req.extensions.get::<InternalHttpDispatch>().is_some() {
            // A cluster coordinator's shard-level write: the primary stage
            // (the coordinating stage is held where the request came in).
            vec![pressure.primary(bytes, req.extensions.get::<CoordinatorLocal>().is_some())?]
        } else {
            // A write received here and executed here.
            let coordinating = pressure.coordinating(bytes)?;
            vec![coordinating, pressure.primary(bytes, true)?]
        }
    } else { Vec::new() };
    let nested = req.extensions.get::<NestedRequest>().is_some();
    ENGINE_POOL.scope(pool, NESTED.scope(nested, READ_SHARD.scope(selected, dispatch_inner(node, req)))).await
}

async fn dispatch_inner(node: Arc<Node>, mut req: Req) -> R {
    let t0 = std::time::Instant::now();
    if let Some(hook) = req.extensions.remove::<AuthorizedDispatch>() {
        return (hook.0)(req).await;
    }
    let warning_owner=req.extensions.get::<input_metadata::Owner>().cloned().ok_or_else(||EsError::internal("HTTP request metadata owner missing"))?;
    // The dispatch descriptor vector and copied index name are URI metadata,
    // not handler payloads. Admit both before constructing either, and keep
    // their charge with the request owner through async handler completion.
    let routing_bytes=req.segs.len().checked_mul(std::mem::size_of::<&str>())
        .and_then(|bytes|bytes.checked_add(req.segs.first().map_or(0,String::len)))
        .ok_or_else(||EsError::new(429,"circuit_breaking_exception","HTTP route metadata overflow"))?;
    warning_owner.0.lock().unwrap().grow(routing_bytes as u64,"HTTP route dispatch metadata")?;
    let mut segs:Vec<&str>=Vec::new();
    segs.try_reserve_exact(req.segs.len()).map_err(|_|EsError::new(429,"circuit_breaking_exception","HTTP route metadata allocation refused"))?;
    segs.extend(req.segs.iter().map(String::as_str));
    let m = req.method.clone();
    let first = segs.first().copied().unwrap_or("");
    observability::record_rest(&req);
    if let Some(e) = deprecations::reject(&req) {
        return Err(e);
    }
    let t_pre = t0.elapsed().as_micros();
    let deprecations = deprecations::for_request(&req,&mut warning_owner.0.lock().unwrap())?;
    elasticsearch_engine::inference::attach(&node.data);
    let t_dep = t0.elapsed().as_micros();
    let r = if segs.is_empty() {
        cluster::root(&node, &req)
    } else if first.starts_with('_') && first != "_all" {
        match first {
            "_search" | "_msearch" if segs.get(1).copied() == Some("template") => query_lang::handle(node, req, None).await,
            "_application" if segs.get(1).copied() == Some("analytics") => analytics::handle(node, req).await,
            "_eql" if segs.get(1).copied() == Some("stats") && segs.len() == 2 && m == Method::GET => misc9::eql_stats(&node),
            "_render" | "_query" | "_sql" | "_eql" | "_rank_eval" | "_application" | "_query_rules" | "_synonyms" => query_lang::handle(node, req, None).await,
            "_search" | "_msearch" | "_count" | "_pit" => search_api::handle(node, req, None).await,
            "_bulk" | "_mget" => docs::handle(node, req, None).await,
            "_mtermvectors" => termvectors::handle(node, req, None).await,
            "_cat" => cat::handle(node, req).await,
            "_tasks" | "_reindex" | "_update_by_query" | "_delete_by_query" => tasks::handle(node, req).await,
            "_async_search" => async_search::handle(node, req, None).await,
            "_field_caps" => async_search::field_caps(node, req, None).await,
            "_data_stream" if ilm::owns_data_stream_route(&req) => ilm::handle_data_stream(node, req).await,
            "_data_stream" => datastream::handle(node, req).await,
            "_snapshot" if segs.get(3).copied() == Some("_clone") && segs.len() == 5 => snapshot_api::handle_clone(node, req).await,
            "_snapshot" => snapshot_api::handle(node, req).await,
            "_ilm" | "_slm" | "_lifecycle" => ilm::handle_global(node, req).await,
            "_watcher" => watcher::handle(node, req).await,
            "_scripts" if segs.get(1).copied() == Some("painless") && segs.get(2).copied() == Some("_context") && segs.len() == 3 && m == Method::GET => misc9::painless_context(&req),
            "_scripts" | "_script_context" | "_script_language" => scripts::handle(node, req).await,
            "_dangling" => misc9::dangling(node, req).await,
            "_streams" => misc9::streams(node, req).await,
            "_nodes" if segs.get(1).copied() == Some("shutdown") && segs.len() == 2 || segs.get(2).copied() == Some("shutdown") && segs.len() == 3 => misc9::shutdown(node, req).await,
            "_nodes" if segs.iter().skip(1).take(2).any(|s| matches!(*s, "stats" | "hot_threads" | "hotthreads" | "usage")) => observability::handle_nodes(node, req).await,
            "_cluster" if matches!(segs.get(1).copied(), Some("stats") | Some("allocation") | Some("reroute")) => observability::handle_cluster(node, req).await,
            "_migration" if segs.get(1).copied() == Some("system_features") => system_features::handle_migration(node, req).await,
            "_migration" if segs.get(1).copied() == Some("reindex") => datastream::migration_reindex(node, req).await,
            "_create_from" => datastream::create_from(node, req).await,
            "_migration" | "_health_report" | "_recovery" | "_shard_stores" | "_search_shards" | "_validate" | "_terms_enum" | "_explain" | "_cache" | "_segments" | "_internal" => observability::handle_global(node, req).await,
            "_resolve" => observability::handle_global(node, req).await,
            "_remote" => observability::handle_global(node, req).await,
            "_ingest" => ingest::handle(node, req).await,
            "_enrich" => enrich::handle(node, req).await,
            "_ccr" => ccr::handle_global(node, req).await,
            "_transform" => transform::handle(node, req).await,
            "_text_structure" => text_structure::handle(node, req).await,
            "_logstash" => logstash::handle(node, req).await,
            "_security" => security::disabled(&req),
            "_inference" => inference::handle(node, req).await,
            "_ml" => ml::handle(node, req).await,
            "_connector" => connector::handle(node, req).await,
            "_fleet" => fleet::handle(node, req, None).await,
            "_monitoring" => monitoring::handle(node, req).await,
            "_profiling" => profiling::handle(node, req).await,
            "_rollup" => rollup::handle_global(node, req).await,
            "_autoscaling" => autoscaling::handle(node, req).await,
            "_rollup_search" => rollup::search(node, req, None).await,
            "_searchable_snapshots" => rollup::searchable_snapshots(node, req, None).await,
            "_license" => license::handle(node, req).await,
            "_xpack" if segs.get(1).copied() != Some("usage") => license::handle(node, req).await,
            "_features" => system_features::handle_features(node, req).await,
            "_nodes" if segs.get(1).copied() == Some("reload_secure_settings") || matches!(segs.get(2).copied(), Some("reload_secure_settings") | Some("_repositories_metering")) => {
                system_features::handle_nodes(node, req).await
            }
            "_ssl" if segs.get(1).copied() == Some("certificates") && segs.len() == 2 => system_features::ssl_certificates(&node, &req),
            "_cluster" | "_nodes" | "_xpack" | "_info" | "_capabilities" => cluster::handle(node, req).await,
            _ => indices::handle_global(node, req).await,
        }
    } else {
        // /{index}/...
        let mut index=String::new();
        index.try_reserve_exact(first.len()).map_err(|_|EsError::new(429,"circuit_breaking_exception","HTTP route index allocation refused"))?;
        index.push_str(first);
        let second = segs.get(1).copied().unwrap_or("");
        match second {
            "" => indices::handle_index(node, req, index).await,
            "_doc" | "_create" | "_update" | "_source" | "_bulk" | "_mget" => docs::handle(node, req, Some(index)).await,
            "_termvectors" | "_mtermvectors" => termvectors::handle(node, req, Some(index)).await,
            "_search" | "_msearch" if segs.get(2).copied() == Some("template") => query_lang::handle(node, req, Some(index)).await,
            "_eql" | "_rank_eval" => query_lang::handle(node, req, Some(index)).await,
            "_search" | "_msearch" | "_count" | "_pit" | "_knn_search" => search_api::handle(node, req, Some(index)).await,
            "_mvt" => mvt::handle(node, req, index).await,
            "_update_by_query" | "_delete_by_query" => tasks::by_query(node, req, index).await,
            "_async_search" => async_search::handle(node, req, Some(index)).await,
            "_field_caps" => async_search::field_caps(node, req, Some(index)).await,
            "_rollover" => datastream::rollover(node, req, index).await,
            "_downsample" => datastream::downsample(node, req, index).await,
            "_graph" => graph::handle(node, req, index).await,
            "_ccr" => ccr::handle_index(node, req, index).await,
            "_fleet" => fleet::handle(node, req, Some(index)).await,
            "_rollup" => rollup::index_caps(node, req, index).await,
            "_rollup_search" => rollup::search(node, req, Some(index)).await,
            "_searchable_snapshots" => rollup::searchable_snapshots(node, req, Some(index)).await,
            "_ilm" | "_lifecycle" | "_clone" | "_shrink" | "_split" | "_freeze" | "_unfreeze" => ilm::handle_index(node, req, index).await,
            "_recovery" | "_segments" | "_shard_stores" | "_disk_usage" | "_field_usage_stats" | "_block" | "_reload_search_analyzers" | "_search_shards" | "_validate" | "_terms_enum" | "_explain" | "_migration" | "_cache" => {
                observability::handle_index(node, req, index).await
            }
            _ => indices::handle_index(node, req, index).await,
        }
    };
    let _ = m;
    elasticsearch_engine::search::search_phases::record_named("engine_dispatch", vec![("pre", t_pre), ("deprecations", t_dep - t_pre), ("handler", t0.elapsed().as_micros() - t_dep)]);
    match r {
        Ok(mut resp) if !deprecations.is_empty() => {
            for d in deprecations {
                let mut memory=warning_owner.0.lock().unwrap();
                let line=response_tree::formatted(&mut memory,format_args!("299 Elasticsearch-{}-{} \"{}\"",elasticsearch_engine::node::ES_VERSION,cluster::BUILD_HASH,WarningEscaped(&d)))?;
                if !resp.headers.iter().any(|(k,v)|k.eq_ignore_ascii_case("warning")&&*v==line){response_tree::push(&mut memory,&mut resp.headers,("Warning",line))?;}

            }
            Ok(resp)
        }
        other => other,
    }
}

struct WarningEscaped<'a>(&'a str);
impl std::fmt::Display for WarningEscaped<'_>{fn fmt(&self,out:&mut std::fmt::Formatter<'_>)->std::fmt::Result{for ch in self.0.chars(){if ch=='\\'||ch=='"'{out.write_str("\\")?;}write!(out,"{ch}")?;}Ok(())}}

/// Render an Elasticsearch error into independently owned, admitted JSON.
/// The temporary error tree remains charged until its encoded body exists.
pub fn render_error_admitted(error:&EsError)->Result<Bytes,EsError>{
    elasticsearch_engine::memory::native_control(||render_error_native(error))
}
fn render_error_native(error: &EsError) -> Result<Bytes, EsError> {
    let mut memory = elasticsearch_engine::memory::breakers().reserve_request(0, "REST error rendering")?;
    let value = error_admitted::render(error, false, &mut memory)?;
    wire::json_admitted(&value, false)
}

fn status_code(s: u16) -> StatusCode {
    StatusCode::from_u16(s).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
}

/// Turn a handler result into the HTTP response.
/// No recursive serialization/admission when the output allocator refuses.
fn encoding_failure(req_head: &ReqHead, error: &EsError) -> Response<Body> {
    let payload: &'static [u8] = if error.status == 429 {
        br#"{"error":{"root_cause":[{"type":"circuit_breaking_exception","reason":"response allocation refused","bytes_wanted":0,"bytes_limit":0,"durability":"PERMANENT"}],"type":"circuit_breaking_exception","reason":"response allocation refused","bytes_wanted":0,"bytes_limit":0,"durability":"PERMANENT"},"status":429}"#
    } else {
        br#"{"error":{"type":"exception","reason":"response encoding failed"},"status":500}"#
    };
    Response::builder().status(if error.status == 429 {429} else {500})
        .header("X-Elastic-Product","Elasticsearch").header("content-type","application/json")
        .body(if req_head.is_head {Body::empty()} else {Body::from(Bytes::from_static(payload))}).unwrap()
}

fn finish(req_head: &ReqHead, r: R) -> Response<Body> {
    // Authentication has already decided this result. A failed response
    // allocation must not replace its challenge with a breaker/encoding error.
    // Only the refusal path uses this bounded static response; normal detailed
    // authentication responses retain their configured challenges and body.
    let unauthorized = match &r { Ok(response) => response.status == 401, Err(error) => error.status == 401 };
    let response = if r.is_err(){elasticsearch_engine::memory::native_control(||finish_inner(req_head,r))}else{finish_inner(req_head,r)};
    if unauthorized && matches!(response.status().as_u16(), 429 | 500) {
        return Response::builder().status(401)
            .header("x-elastic-product", "Elasticsearch")
            .header("content-type", "application/json")
            .header("www-authenticate", "Basic realm=\"security\", charset=\"UTF-8\"")
            .body(if req_head.is_head { Body::empty() } else { Body::from(Bytes::from_static(br#"{"error":{"type":"security_exception","reason":"authentication required"},"status":401}"#)) }).unwrap();
    }
    response
}

fn finish_inner(req_head: &ReqHead, r: R) -> Response<Body> {
    let mut transform_memory=match elasticsearch_engine::memory::breakers().reserve_request(0,"HTTP response transform") { Ok(owner)=>owner,Err(error)=>return encoding_failure(req_head,&error) };
    let retained_owner; // declared first, so headers drop before this owner on every exit
    let (status, mut body, mut extra) = match r {
        Ok(resp) => (resp.status, resp.body, resp.headers),
        Err(e) if !req_head.detailed_errors => match error_admitted::non_detailed(&e,req_head.api_version,&mut transform_memory){Ok(rendered)=>rendered,Err(error)=>return encoding_failure(req_head,&error)},
        Err(e) => {
            let v = match error_admitted::render(&e,req_head.error_trace,&mut transform_memory){Ok(value)=>value,Err(error)=>return encoding_failure(req_head,&error)};
            (e.status, RespBody::Json(v), Vec::new())
        }
    };
    retained_owner = match &body {
        RespBody::Owned { bytes, .. } => Some(bytes.clone()),
        RespBody::RetainedJson { owner, .. } => Some(owner.clone()),
        _ => None,
    };
    if let RespBody::RetainedJson { value, .. } = body { body = RespBody::Json(value); }
    // Keep the source owner even if pretty/FLS/wire transforms replace body.
    let _ = &retained_owner;
    for w in &req_head.route_warnings {
        let line = match response_tree::formatted(&mut transform_memory,format_args!("299 Elasticsearch-{}-{} \"{w}\"", elasticsearch_engine::node::ES_VERSION, cluster::BUILD_HASH)){Ok(value)=>value,Err(error)=>return encoding_failure(req_head,&error)};
        if !extra.iter().any(|(k, v)| k.eq_ignore_ascii_case("warning") && *v == line) {
            if let Err(error)=response_tree::push(&mut transform_memory,&mut extra,("Warning",line)){return encoding_failure(req_head,&error)}
            extra.rotate_right(1);
        }
    }
    let mut content_type = match response_tree::formatted(&mut transform_memory,&req_head.content_type){Ok(value)=>value,Err(error)=>return encoding_failure(req_head,&error)};
    if req_head.pretty || req_head.filter_path.is_some() || req_head.human || req_head.wire.is_some() {
        let raw = match &body {
            RespBody::Raw(s) => Some(s.as_str()),
            RespBody::Owned { bytes, json: true } => std::str::from_utf8(bytes).ok(),
            _ => None,
        };
        if let Some(s) = raw {
            match response_tree::parse(s, &mut transform_memory) {
                Ok(value)=>body=RespBody::Json(value),
                Err(error) if error.status==429=>return encoding_failure(req_head,&error),
                Err(_)=>match response_tree::parse_source(s,&mut transform_memory) {
                Ok(mut source)=>{
                if let Some(fp)=&req_head.filter_path {source=match response_tree::filter_source(&source,fp,&mut transform_memory){Ok(value)=>value,Err(error)=>return encoding_failure(req_head,&error)};}
                body = match wire::json_admitted(&source, req_head.pretty) {
                    Ok(bytes) => RespBody::Owned { bytes, json: true },
                    Err(error) => return encoding_failure(req_head, &error),
                };
                },
                Err(error) if error.status==429=>return encoding_failure(req_head,&error),
                Err(_)=>{},
                }
            }
        }
    }
    if req_head.typed && !req_head.is_head && !req_head.human && req_head.filter_path.is_none() && req_head.wire.is_none() && !req_head.pretty && !req_head.gzip {
        // The headers the HTTP response below would carry.
        let typed = matches!(body, RespBody::Raw(_)).then(|| {
            let ct=extra.iter().find(|(k,_)|k.eq_ignore_ascii_case("content-type")).map(|(_,v)|v.as_str()).unwrap_or(&content_type);
            typed_headers(extra.iter().filter(|(k,_)|!k.eq_ignore_ascii_case("content-type")&&!k.eq_ignore_ascii_case("x-elastic-product")).map(|(k,v)|(*k,v.as_str()))
                .chain([("x-elastic-product","Elasticsearch"),("content-type",ct)]))
        }).flatten();
        if let (Some(headers), RespBody::Raw(text)) = (typed, &mut body) {
            let status = status_code(status);
            let mut resp = Response::new(Body::empty());
            *resp.status_mut() = status;
            resp.extensions_mut().insert(TypedResponse { status: status.as_u16(), headers, body: std::mem::take(text) });
            return resp;
        }
    }
    if req_head.in_process && !req_head.is_head && !req_head.human && req_head.filter_path.is_none() && req_head.wire.is_none() && !req_head.pretty {
        if let RespBody::Json(v) = body {
            let ct=extra.iter().find(|(k,_)|k.eq_ignore_ascii_case("content-type")).map(|(_,v)|v.as_str()).unwrap_or(&content_type);
            let headers=extra.iter().filter(|(k,_)|!k.eq_ignore_ascii_case("content-type")&&!k.eq_ignore_ascii_case("x-elastic-product")).map(|(k,v)|(*k,v.as_str()));
            let mut resp=match outgoing_headers::response(status,headers.chain([("x-elastic-product","Elasticsearch"),("content-type",ct)].into_iter()),Body::empty()) {
                Ok(response)=>response,Err(error)=>return encoding_failure(req_head,&error),
            };
            resp.extensions_mut().insert(JsonResponse(v));
            return resp;
        }
    }
    let bytes: Bytes = match body {
        RespBody::Json(mut v) | RespBody::RetainedJson { value: mut v, .. } => {
            if req_head.human {
                if let Err(error)=response_tree::humanize(&mut v,&mut transform_memory){return encoding_failure(req_head,&error)}
            }
            if let Some(fp) = &req_head.filter_path {
                v = match response_tree::filter(&v,fp,&mut transform_memory){Ok(value)=>value,Err(error)=>return encoding_failure(req_head,&error)};
            }
            let encoded = if let Some(w) = req_head.wire {
                content_type = match response_tree::formatted(&mut transform_memory,w.content_type()){Ok(value)=>value,Err(error)=>return encoding_failure(req_head,&error)};
                wire::encode_admitted(w, &v)
            } else {
                wire::json_admitted(&v, req_head.pretty)
            };
            match encoded {
                Ok(bytes) => bytes,
                Err(error) => {
                    return encoding_failure(req_head, &error);
                }
            }
        },
        RespBody::Raw(s) => Bytes::from(s),
        RespBody::Text(s) => {
            content_type = match response_tree::formatted(&mut transform_memory,"text/plain; charset=UTF-8"){Ok(value)=>value,Err(error)=>return encoding_failure(req_head,&error)};
            Bytes::from(s)
        }
        RespBody::Bytes(b) => Bytes::from(b),
        RespBody::Owned { bytes, .. } => bytes,
        RespBody::Empty => Bytes::new(),
    };
    if let Some((_, ct)) = extra.iter().find(|(k, _)| k.eq_ignore_ascii_case("content-type")) {
        content_type = match response_tree::formatted(&mut transform_memory,ct){Ok(value)=>value,Err(error)=>return encoding_failure(req_head,&error)};
    }
    let length=match response_tree::formatted(&mut transform_memory,bytes.len()){Ok(value)=>value,Err(error)=>return encoding_failure(req_head,&error)};
    let gzip=if req_head.gzip && !req_head.is_head && bytes.len()>64 { compression::gzip_body(&bytes).ok() } else {None};
    let extra_headers=extra.iter().filter(|(k,_)| !k.eq_ignore_ascii_case("content-type") && !k.eq_ignore_ascii_case("x-elastic-product")).map(|(k,v)|(*k,v.as_str()));
    let fixed=[Some(("x-elastic-product","Elasticsearch")),Some(("content-type",content_type.as_str())),
        req_head.is_head.then_some(("content-length",length.as_str())),
        gzip.as_ref().map(|_|("content-encoding","gzip")),gzip.as_ref().map(|_|("vary","Accept-Encoding"))];
    let body=if req_head.is_head {Body::empty()} else {gzip.unwrap_or_else(||Body::from(bytes))};
    match outgoing_headers::response(status,extra_headers.chain(fixed.into_iter().flatten()),body) {
        Ok(response)=>response, Err(error)=>encoding_failure(req_head,&error),
    }
}

/// `http.detailed_errors.enabled: false` (`RestResponse` /
/// `ElasticsearchException.generateFailureXContent`): ES 9 renders only the
/// error's type and reason as an object; `compatible-with=8` keeps 8.x's
/// `"error": "SimpleClassName[reason]"` string, with a deprecation warning.
fn non_detailed_error(e: &EsError, api_version: RestApiVersion) -> (u16, RespBody, Vec<(&'static str, String)>) {
    if api_version.is_v8() {
        // `getExceptionName` in reverse: `index_not_found_exception` -> `IndexNotFoundException`
        // (the `elasticsearch_` prefix ES strips from the type is part of the class name).
        let base = match e.kind.as_str() {
            "exception" | "timeout_exception" | "status_exception" | "parse_exception" | "security_exception" | "generation_exception" => format!("elasticsearch_{}", e.kind),
            _ => e.kind.clone(),
        };
        let class: String = base.split('_').map(|w| {
            let mut c = w.chars();
            c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
        }).collect();
        let warning = format!(
            "299 Elasticsearch-{}-{} \"The JSON format of non-detailed errors has changed in Elasticsearch 9.0 to match the JSON structure used for detailed errors.\"",
            elasticsearch_engine::node::ES_VERSION,
            cluster::BUILD_HASH
        );
        (e.status, RespBody::Json(serde_json::json!({"error": format!("{class}[{}]", e.reason), "status": e.status})), vec![("Warning", warning)])
    } else {
        (e.status, RespBody::Json(serde_json::json!({"error": {"type": e.kind, "reason": e.reason}, "status": e.status})), Vec::new())
    }
}

struct ReqHead {
    /// `http.detailed_errors.enabled` (default true).
    detailed_errors: bool,
    api_version: RestApiVersion,
    pretty: bool,
    filter_path: Option<String>,
    error_trace: bool,
    is_head: bool,
    gzip: bool,
    content_type: String,
    /// `?human` on a stats-style API.
    human: bool,
    /// `?format=yaml|cbor|smile` / matching `Accept`.
    wire: Option<wire::Wire>,
    in_process: bool,
    /// [`InProcessTyped`].
    typed: bool,
    /// Warnings of a deprecated route (`Route.deprecatedForRemoval`): ES's
    /// `DeprecationRestHandler` logs them before the handler runs, so error
    /// responses carry them too.
    route_warnings: Vec<String>,
    _metadata: Option<std::sync::Arc<std::sync::Mutex<elasticsearch_engine::memory::Reservation>>>,
}

/// The non-JSON response encoding a request negotiates (`?format`, else
/// `Accept`, else the body's type) and its `Content-Type`; `None` for JSON
/// or when negotiation fails (the handler then reports the media type error).
#[path="media_admitted.rs"]
mod media_admitted;
/// Fallible edge variant; caller retains the owner through the selected MIME.
pub fn negotiate_wire_admitted(headers:&HeaderMap,format:Option<&str>,has_content:bool,memory:&mut elasticsearch_engine::memory::Reservation)->Result<Option<(wire::Wire,String)>,EsError>{
    media_admitted::select(headers,format,has_content,memory)
}

pub fn negotiate_wire(headers: &HeaderMap, format: Option<&str>, has_content: bool) -> Option<(wire::Wire, String)> {
    let sel = media_select(headers, format, has_content).ok()?;
    sel.wire.map(|w| (w, sel.content_type))
}

/// Request body media types ES accepts (`application/x-www-form-urlencoded`,
/// `text/plain`, ... are rejected with 406).
fn body_type_supported(ct: &str) -> bool {
    let mime=ct.split(';').next().unwrap_or("").trim();
    ["application/json","application/x-ndjson","application/smile","application/cbor","application/yaml",
     "application/vnd.elasticsearch+json","application/vnd.elasticsearch+x-ndjson","application/vnd.elasticsearch+smile",
     "application/vnd.elasticsearch+cbor","application/vnd.elasticsearch+yaml"].iter().any(|supported|mime.eq_ignore_ascii_case(supported))
}

fn hex_pair(first: u8, second: u8) -> Option<u8> {
    fn digit(byte: u8) -> Option<u8> {
        match byte { b'0'..=b'9' => Some(byte-b'0'), b'a'..=b'f' => Some(byte-b'a'+10), b'A'..=b'F' => Some(byte-b'A'+10), _ => None }
    }
    Some((digit(first)? << 4) | digit(second)?)
}

fn decode_component(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Some(v) = hex_pair(b[i + 1], b[i + 2]) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        if b[i] == b'+' {
            out.push(b' ');
        } else {
            out.push(b[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn decode_path_segment(s: &str) -> String {
    // Path segments: '+' stays literal.
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Some(v) = hex_pair(b[i + 1], b[i + 2]) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// ES `ParsedMediaType`: one media type with its (lowercased) parameters.
/// A header listing several media ranges (`,`) or `*/*` parses to `None`,
/// as in ES.
struct ParsedMediaType {
    original: String,
    /// `type/subtype`, lowercased.
    mime: String,
    params: Vec<(String, String)>,
}

/// RFC 7230 `tchar`.
fn is_token(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
}

/// ES `ParsedMediaType.parseMediaType`; `Err` is the `illegal_argument_exception` reason.
fn parse_media_type(h: &str) -> Result<Option<ParsedMediaType>, String> {
    if h.contains(',') || h == "*/*" {
        return Ok(None);
    }
    let lower = h.to_lowercase();
    let elements: Vec<&str> = lower.split(';').collect();
    let tm: Vec<&str> = elements[0].split('/').collect();
    if !(tm.len() == 2 && is_token(tm[0].trim()) && is_token(tm[1].trim())) {
        return Err(format!("invalid media-type [{h}]"));
    }
    let mut params = Vec::new();
    for e in &elements[1..] {
        let e = e.trim();
        if e.is_empty() {
            continue;
        }
        let kv: Vec<&str> = e.split('=').collect();
        let bad = |x: &str, end: bool| x.is_empty() || if end { x.ends_with(char::is_whitespace) } else { x.starts_with(char::is_whitespace) };
        if kv.len() != 2 || bad(kv[0], true) || bad(kv[1], false) {
            return Err(format!("invalid parameters for header [{h}]"));
        }
        params.push((kv[0].trim().to_string(), kv[1].trim().to_string()));
    }
    Ok(Some(ParsedMediaType { original: h.to_string(), mime: format!("{}/{}", tm[0].trim(), tm[1].trim()), params }))
}

impl ParsedMediaType {
    fn param(&self, k: &str) -> Option<&str> {
        self.params.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
    }
}

/// ES `XContentType.fromMediaType`: the canonical response media type and
/// wire format of a parsed media type (`None`: not an x-content type).
fn xcontent_of(mime: &str) -> Option<(&'static str, Option<wire::Wire>)> {
    Some(match mime {
        "application/json" | "application/x-ndjson" | "application/*" => ("application/json", None),
        "application/vnd.elasticsearch+json" | "application/vnd.elasticsearch+x-ndjson" => ("application/vnd.elasticsearch+json", None),
        "application/yaml" => ("application/yaml", Some(wire::Wire::Yaml)),
        "application/vnd.elasticsearch+yaml" => ("application/vnd.elasticsearch+yaml", Some(wire::Wire::Yaml)),
        "application/smile" => ("application/smile", Some(wire::Wire::Smile)),
        "application/vnd.elasticsearch+smile" => ("application/vnd.elasticsearch+smile", Some(wire::Wire::Smile)),
        "application/cbor" => ("application/cbor", Some(wire::Wire::Cbor)),
        "application/vnd.elasticsearch+cbor" => ("application/vnd.elasticsearch+cbor", Some(wire::Wire::Cbor)),
        _ => return None,
    })
}

/// What the `Accept` / `Content-Type` headers select for one request.
#[derive(Debug)]
struct MediaSel {
    api_version: RestApiVersion,
    /// Canonical response media type of a structured (JSON/YAML/...) body.
    content_type: String,
    wire: Option<wire::Wire>,
}

fn media_type_exception(headers: &str, cause: EsError) -> EsError {
    EsError::new(400, "media_type_header_exception", format!("Invalid media-type value on headers [{headers}]")).caused_by(cause)
}

/// ES `RestRequest` header parsing + `RestCompatibleVersionHelper` +
/// `AbstractRestChannel.newBuilder`'s response type choice: the request's
/// REST API version (`compatible-with=8` selects V8; 7 or anything but the
/// current / previous major is rejected) and the response content type
/// (`?format`, else `Accept`, else the request body's type, else JSON; the
/// `Accept` parameters, e.g. `compatible-with=8`, are echoed).
fn media_select(headers: &HeaderMap, format: Option<&str>, has_content: bool) -> Result<MediaSel, EsError> {
    let get = |n: &str| headers.get(n).map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned());
    let accept = match get("accept").map(|h| parse_media_type(&h)) {
        None => None,
        Some(Ok(p)) => p,
        Some(Err(e)) => return Err(media_type_exception("Accept", EsError::illegal_argument(e))),
    };
    let ctype = match get("content-type").map(|h| parse_media_type(&h)) {
        None => None,
        Some(Ok(p)) => p,
        Some(Err(e)) => return Err(media_type_exception("Content-Type", EsError::illegal_argument(e))),
    };
    let version_err = |reason: String| media_type_exception("Content-Type, Accept", EsError::new(400, "status_exception", reason));
    let parse_version = |p: &Option<ParsedMediaType>| -> Result<Option<i16>, EsError> {
        match p.as_ref().and_then(|p| p.param("compatible-with")) {
            None => Ok(None),
            Some(v) => v.parse::<i8>().map(|n| Some(n as i16)).map_err(|_| {
                media_type_exception("Content-Type, Accept", EsError::new(400, "number_format_exception", format!("For input string: \"{v}\"")))
            }),
        }
    };
    let (cur, min) = (RestApiVersion::CURRENT.major() as i16, RestApiVersion::MINIMUM_SUPPORTED.major() as i16);
    let show = |p: &Option<ParsedMediaType>| p.as_ref().map(|p| p.original.clone()).unwrap_or_else(|| "null".into());
    let a_version = parse_version(&accept)?;
    let c_version = parse_version(&ctype)?;
    let accept_version = a_version.unwrap_or(cur);
    let ctype_version = c_version.unwrap_or(cur);
    if accept_version > cur || accept_version < min {
        return Err(version_err(format!("Accept version must be either version {cur} or {min}, but found {accept_version}. Accept={}", show(&accept))));
    }
    let mut api_version = RestApiVersion::CURRENT;
    if has_content {
        if ctype_version > cur || ctype_version < min {
            return Err(version_err(format!(
                "Content-Type version must be either version {cur} or {min}, but found {ctype_version}. Content-Type={}",
                show(&ctype)
            )));
        }
        if ctype_version != accept_version {
            return Err(version_err(format!(
                "A compatible version is required on both Content-Type and Accept headers if either one has requested a compatible version and the compatible versions must match. Accept={}, Content-Type={}",
                show(&accept),
                show(&ctype)
            )));
        }
        if c_version.is_none() != a_version.is_none() {
            return Err(version_err(format!(
                "A compatible version is required on both Content-Type and Accept headers if either one has requested a compatible version. Accept={}, Content-Type={}",
                show(&accept),
                show(&ctype)
            )));
        }
        if ctype_version < cur {
            api_version = RestApiVersion::MINIMUM_SUPPORTED;
        }
    }
    if accept_version < cur {
        api_version = RestApiVersion::MINIMUM_SUPPORTED;
    }
    // Response type: `?format`, else `Accept`, else the request body's type, else JSON.
    let from_format = match format {
        Some("json") => Some(("application/json", None)),
        Some("yaml") => Some(("application/yaml", Some(wire::Wire::Yaml))),
        Some("smile") => Some(("application/smile", Some(wire::Wire::Smile))),
        Some("cbor") => Some(("application/cbor", Some(wire::Wire::Cbor))),
        _ => None,
    };
    let (base, wire) = from_format
        .or_else(|| accept.as_ref().and_then(|a| xcontent_of(&a.mime)))
        .or_else(|| ctype.as_ref().and_then(|c| xcontent_of(&c.mime)))
        .unwrap_or(("application/json", None));
    let mut content_type = base.to_string();
    if let Some(a) = &accept {
        for (k, v) in &a.params {
            content_type.push(';');
            content_type.push_str(k);
            content_type.push('=');
            content_type.push_str(v);
        }
    }
    Ok(MediaSel { api_version, content_type, wire })
}

/// How ES's `RestController` holds a REST request's body on the
/// `inflight_requests` breaker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InFlight {
    /// Charged, and may trip.
    Trips,
    /// Charged without tripping (`BaseRestHandler.canTripCircuitBreaker()
    /// == false`): requests that must get through a node under memory
    /// pressure, to inspect it or raise its limits.
    Holds,
    /// A streamed `_bulk`: charged 0 bytes, but checked (ES 9
    /// `RestController` calls `addEstimateBytesAndMaybeBreak(0)` for streamed
    /// content, so the parent can still trip it); indexing pressure accounts
    /// its bytes.
    Streamed,
    /// Not charged nor checked: a cluster's internal dispatch of a body its
    /// client edge already holds (ES charges a body once per node).
    Uncharged,
}

/// The [`InFlight`] policy of `method path`: ES 9's
/// `canTripCircuitBreaker() == false` handlers hold, `_bulk` is streamed
/// unless `rest.incremental_bulk` is off (then aggregated and charged).
pub fn in_flight_policy(method: &str, path: &str, incremental_bulk: bool) -> InFlight {
    // Classify before URI allocation/admission. Six borrowed slots distinguish
    // every exact route below; longer prefixes only match explicit tails.
    let mut first = [""; 6];
    let mut count = 0usize;
    let mut last = None;
    for segment in path.split('?').next().unwrap_or("").split('/').filter(|s| !s.is_empty()) {
        if count < first.len() { first[count] = segment; count += 1; }
        last = Some(segment);
    }
    let segs = &first[..count];
    let read = matches!(method, "GET" | "HEAD");
    let holds = match segs {
        // RestMainAction (the cat actions trip: AbstractCatAction keeps the
        // default).
        [] => true,
        // RestListTasksAction, RestCancelTasksAction (RestGetTaskAction,
        // `GET _tasks/{id}`, trips).
        ["_tasks"] | ["_tasks", "_cancel"] | ["_tasks", _, "_cancel"] => true,
        // Cluster health / settings / state / stats, voting config
        // exclusions (add and clear), allocation explain.
        ["_cluster", "health" | "settings" | "state" | "stats" | "voting_config_exclusions", ..] => true,
        ["_cluster", "allocation", "explain"] => true,
        // RestGetHealthAction, RestRemoteClusterInfoAction,
        // RestNodesCapabilitiesAction, RestPrevalidateNodeRemovalAction.
        ["_health_report", ..] | ["_remote", "info"] | ["_capabilities"] | ["_internal", "_prevalidate_node_removal"] => true,
        // RestReloadSecureSettingsAction, RestPut/DeleteShutdownNodeAction
        // (RestGetShutdownStatusAction trips).
        ["_nodes", "reload_secure_settings"] | ["_nodes", _, "reload_secure_settings"] => true,
        ["_nodes", _, "shutdown"] => matches!(method, "PUT" | "DELETE"),
        // The nodes info / stats / usage / hot threads actions (GET only;
        // `_nodes/shutdown`, `_repositories_metering` trip).
        ["_nodes", rest @ ..] if read => nodes_read_exempt(rest),
        // RestIndicesStatsAction (`_stats`, `{index}/_stats`; not the
        // `_data_stream`, `_enrich`, `_transform` stats), RestClearIndicesCacheAction.
        ["_stats", ..] | ["_cache", "clear"] => true,
        [index, "_stats", ..] | [index, "_cache", "clear"] if !index.starts_with('_') || *index == "_all" => true,
        _ => false,
    };
    if holds {
        InFlight::Holds
    } else if incremental_bulk && last == Some("_bulk") {
        InFlight::Streamed
    } else {
        InFlight::Trips
    }
}

#[cfg(test)]
mod ingress_policy_tests {
    use super::{in_flight_policy,InFlight};
    #[test]
    fn borrowed_prefix_classifier_preserves_long_and_exact_routes() {
        for path in ["/_nodes", "/_nodes/_local/stats/breaker", "/_cluster/health/a/b/c/d/e", "/index/_stats/a/b/c/d/e"] {
            assert_eq!(in_flight_policy("GET",path,true),InFlight::Holds,"{path}");
        }
        for path in ["/_nodes/a/b/c/d/e/f", "/_tasks/a/_cancel/extra", "/_nodes/shutdown", "/_cat/indices"] {
            assert_eq!(in_flight_policy("GET",path,true),InFlight::Trips,"{path}");
        }
        assert_eq!(in_flight_policy("POST","/a/b/c/d/e/f/g/_bulk?refresh=true",true),InFlight::Streamed);
        assert_eq!(in_flight_policy("POST","/a/b/c/d/e/f/g/_bulk",false),InFlight::Trips);
    }
}

/// `GET _nodes/...` paths whose ES actions do not trip: nodes info
/// (`_nodes`, `{id}`, `{metric}`, `{id}/{metric}`), stats, usage, hot threads.
fn nodes_read_exempt(rest: &[&str]) -> bool {
    let tripping = |s: &&str| matches!(*s, "shutdown" | "_repositories_metering");
    !rest.iter().any(tripping) && rest.len() <= 4
}

/// `rest.incremental_bulk` (dynamic, default true): `_bulk` bodies are
/// streamed (not charged to `inflight_requests`).
pub fn incremental_bulk(node: &Node) -> bool {
    const KEY: &str = "rest.incremental_bulk";
    let text = |v: &Value| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
    let cs = node.cluster_settings.read();
    let v = cs.1.get(KEY).or(cs.0.get(KEY)).map(text).or_else(|| node.node_settings.read().get(KEY).map(text));
    v.is_none_or(|v| v.trim() != "false")
}

/// A request body held on `inflight_requests` until dropped (after the
/// response is built).
pub struct InFlightBody {
    reservation: elasticsearch_engine::memory::Reservation,
    policy: InFlight,
    buffered: Option<elasticsearch_engine::memory::Reservation>,
    breakers: &'static elasticsearch_engine::memory::Breakers,
}

impl InFlightBody {
    const LABEL: &'static str = "<http_request>";

    /// Hold `bytes` under `policy`. A trip is reported only for a request
    /// that `authenticated()` (ES authenticates first): an unauthenticated
    /// one is held without tripping and gets its 401 from the
    /// authentication that follows.
    pub fn reserve(breakers: &'static elasticsearch_engine::memory::Breakers, policy: InFlight, bytes: u64, authenticated: impl FnOnce() -> bool) -> Result<InFlightBody, EsError> {
        use elasticsearch_engine::memory::Kind;
        let reservation = match policy {
            InFlight::Trips => match breakers.reserve(Kind::InflightRequests, bytes, Self::LABEL) {
                Ok(r) => r,
                Err(_) if !authenticated() => return Ok(InFlightBody { reservation: breakers.reserve_without_breaking(Kind::InflightRequests, bytes), policy: InFlight::Holds, buffered:None, breakers }),
                Err(e) => return Err(e),
            },
            InFlight::Holds => breakers.reserve_without_breaking(Kind::InflightRequests, bytes),
            InFlight::Streamed => match breakers.reserve(Kind::InflightRequests, 0, Self::LABEL) {
                Ok(r) => r,
                Err(_) if !authenticated() => return Ok(InFlightBody { reservation: breakers.reserve_without_breaking(Kind::InflightRequests, 0), policy: InFlight::Uncharged, buffered:None, breakers }),
                Err(e) => return Err(e),
            },
            InFlight::Uncharged => breakers.reserve_without_breaking(Kind::InflightRequests, 0),
        };
        Ok(InFlightBody { reservation, policy, buffered:None, breakers })
    }

    /// Admit aggregate buffering before growing an owned body buffer. A
    /// streamed/internal request keeps its REST inflight semantics, but its
    /// actual buffering copy still occupies the request memory budget.
    fn buffering(&mut self, bytes:u64, authenticated:bool)->Result<(),EsError> {
        use elasticsearch_engine::memory::Kind;
        // REST inflight accounting is the logical body length (with its ES
        // overhead), not allocator capacity or the transient realloc peak.
        // Admit the latter separately so a sub-limit body is not rejected by
        // the inflight limit merely because the transport split it in chunks.
        if self.buffered.is_none() {
            self.buffered=Some(if self.policy==InFlight::Holds||!authenticated {
                self.breakers.reserve_native(0,"HTTP control body")?
            } else { self.breakers.reserve(Kind::Request,0,Self::LABEL)? });
        }
        let held=self.buffered.as_mut().unwrap();
        if bytes<=held.bytes(){held.shrink(held.bytes()-bytes);return Ok(());}
        let more=bytes-held.bytes();
        held.grow(more,Self::LABEL)
    }

    /// The body was decoded (gzip): it holds `decoded` bytes now, not the
    /// compressed ones (ES charges the decompressed length).
    pub fn decoded(&mut self, decoded: u64) -> Result<(), EsError> {
        let held = self.reservation.bytes();
        match self.policy {
            InFlight::Streamed | InFlight::Uncharged => Ok(()),
            InFlight::Holds => {
                self.reservation.shrink(held);
                self.reservation.grow_without_breaking(decoded);
                Ok(())
            }
            InFlight::Trips => {
                self.reservation.shrink(held);
                self.reservation.grow(decoded, Self::LABEL)
            }
        }
    }
}

// Keep old and new capacity charged during reallocations. `base` covers
// compressed input and decoder scratch while the output is being expanded.
fn append_body(out:&mut Vec<u8>,chunk:&[u8],limit:usize,base:u64,budget:&mut InFlightBody,authenticated:bool)->Result<(),EsError> {
    let needed=out.len().checked_add(chunk.len()).filter(|n|*n<=limit)
        .ok_or_else(||EsError::new(413,"content_too_long_exception","request body too large"))?;
    if !authenticated && budget.policy==InFlight::Trips { budget.policy=InFlight::Holds; }
    budget.decoded(needed as u64)?;
    if needed>out.capacity() {
        let capacity=needed.max(out.capacity().saturating_mul(2)).max(4096).min(limit);
        budget.buffering(base.saturating_add(out.capacity() as u64).saturating_add(capacity as u64),authenticated)?;
        out.try_reserve_exact(capacity-out.len()).map_err(|_|EsError::internal("unable to allocate request body"))?;
    }
    budget.buffering(base.saturating_add(out.capacity() as u64),authenticated)?;
    out.extend_from_slice(chunk);
    Ok(())
}
pub async fn collect_request_body(body:Body,limit:usize,budget:&mut InFlightBody,authenticated:bool)->Result<Vec<u8>,EsError> {
    use futures::StreamExt;
    let mut stream=body.into_data_stream();
    let mut out=Vec::new();
    while let Some(chunk)=stream.next().await {
        let chunk=chunk.map_err(|e|EsError::new(400,"parse_exception",format!("failed to read request body: {e}")))?;
        append_body(&mut out,&chunk,limit,chunk.len() as u64,budget,authenticated)?;
    }
    budget.buffering(out.capacity() as u64,authenticated)?;
    Ok(out)
}
pub fn decode_request_body_guarded(raw:Vec<u8>,gzip:bool,limit:usize,budget:&mut InFlightBody,authenticated:bool)->Result<Vec<u8>,EsError> {
    if !gzip{return Ok(raw);}
    use std::io::Read;
    // flate2 owns a bounded DEFLATE window and tables even before read().
    let held=budget.buffered.as_ref().map_or(budget.reservation.bytes(),|r|r.bytes());
    let base=held.max(raw.len() as u64).saturating_add(128<<10);
    budget.buffering(base,authenticated)?;
    let mut decoder=flate2::read::MultiGzDecoder::new(&raw[..]);
    let mut out=Vec::new();let mut chunk=[0u8;8192];
    loop {
        let n=decoder.read(&mut chunk).map_err(|_|EsError::parsing("failed to decompress gzip request body"))?;
        if n==0{break;}
        append_body(&mut out,&chunk[..n],limit,base,budget,authenticated)?;
    }
    drop(decoder);drop(raw);
    budget.buffering(out.capacity() as u64,authenticated)?;
    Ok(out)
}
#[cfg(test)]
fn decode_request_body(raw:Bytes,gzip:bool,limit:usize)->Result<Bytes,EsError> {
    let mut budget=InFlightBody::reserve(elasticsearch_engine::memory::breakers(),InFlight::Holds,0,||true)?;
    decode_request_body_guarded(raw.to_vec(),gzip,limit,&mut budget,true).map(Bytes::from)
}

#[cfg(test)]
mod request_body_limit_tests {
    use super::*;
    fn gzip(bytes: &[u8]) -> Bytes {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(bytes).unwrap();
        Bytes::from(encoder.finish().unwrap())
    }
    #[test]
    fn body_growth_and_gzip_expansion_are_admitted_before_buffer_growth() {
        use elasticsearch_engine::memory::{Breakers,Kind};
        let breakers=Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "network.breaker.inflight_requests.limit"=>Some("256kb".into()),
            "indices.breaker.request.limit"=>Some("256kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        let mut budget=InFlightBody::reserve(breakers,InFlight::Trips,0,||true).unwrap();
        let mut out=Vec::new();
        let error=append_body(&mut out,&vec![1;200000],1<<20,0,&mut budget,true).unwrap_err();
        assert_eq!(error.status,429);
        assert_eq!(out.capacity(),0,"admission must precede allocation");
        drop(budget);
        assert_eq!(breakers.get(Kind::InflightRequests).reserved(),0);
        let mut budget=InFlightBody::reserve(breakers,InFlight::Trips,0,||true).unwrap();
        let error=decode_request_body_guarded(gzip(&vec![0;200000]).to_vec(),true,1<<20,&mut budget,true).unwrap_err();
        assert_eq!(error.status,429,"compressed size cannot bypass aggregate limit");
        drop(budget);
        assert_eq!(breakers.get(Kind::InflightRequests).reserved(),0);
    }

    #[tokio::test]
    async fn inflight_counts_body_length_not_chunk_or_reallocation_capacity() {
        use elasticsearch_engine::memory::{Breakers,Kind};
        let breakers=Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "network.breaker.inflight_requests.limit"=>Some("1mb".into()),
            "indices.breaker.request.limit"=>Some("8mb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        let chunks=futures::stream::iter((0..300).map(|_|Ok::<_,std::io::Error>(Bytes::from(vec![0;1024]))));
        let mut budget=InFlightBody::reserve(breakers,InFlight::Trips,0,||true).unwrap();
        let body=collect_request_body(Body::from_stream(chunks),1<<20,&mut budget,true).await.unwrap();
        assert_eq!(body.len(),300*1024);
        assert_eq!(breakers.get(Kind::InflightRequests).reserved(),body.len() as u64);
        assert!(breakers.get(Kind::Request).reserved()>=body.capacity() as u64);
        drop(body);drop(budget);
        assert_eq!(breakers.get(Kind::InflightRequests).reserved(),0);
        assert_eq!(breakers.get(Kind::Request).reserved(),0);
    }

    #[tokio::test]
    async fn unauthenticated_chunked_body_keeps_authentication_precedence() {
        use elasticsearch_engine::memory::{Breakers,Kind};
        let breakers=Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "network.breaker.inflight_requests.limit"=>Some("1kb".into()),
            "indices.breaker.request.limit"=>Some("1kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        let mut budget=InFlightBody::reserve(breakers,InFlight::Trips,0,||false).unwrap();
        let body=collect_request_body(Body::from(vec![0;8192]),1<<20,&mut budget,false).await.unwrap();
        assert_eq!(body.len(),8192);
        assert_eq!(breakers.get(Kind::InflightRequests).reserved(),8192);
        drop(body);drop(budget);
        assert_eq!(breakers.get(Kind::InflightRequests).reserved(),0);
        assert_eq!(breakers.get(Kind::Request).reserved(),0);
    }

    #[tokio::test]
    async fn chunked_body_admission_applies_without_content_length() {
        use elasticsearch_engine::memory::Breakers;
        let breakers=Breakers::leak_for_tests(1<<30);
        breakers.apply_settings(&|key|match key {
            "network.breaker.inflight_requests.limit"=>Some("64kb".into()),
            "indices.breaker.total.limit"=>Some("-1".into()),_=>None,
        });
        let chunks=futures::stream::iter((0..8).map(|_|Ok::<_,std::io::Error>(Bytes::from(vec![0;8192]))));
        let mut budget=InFlightBody::reserve(breakers,InFlight::Trips,0,||true).unwrap();
        let error=collect_request_body(Body::from_stream(chunks),1<<20,&mut budget,true).await.unwrap_err();
        assert_eq!(error.status,429);
        drop(budget);
        assert_eq!(breakers.inflight_requests.reserved(),0);
    }

    #[test]
    fn compressed_body_respects_decoded_boundary() {
        assert_eq!(decode_request_body(gzip(&[42; 1024]), true, 1024).unwrap().len(), 1024);
        assert!(decode_request_body(gzip(&[42; 1025]), true, 1024).is_err());
        assert!(decode_request_body(gzip(&[0; 1_000_000]), true, 1024).is_err());
    }
    #[test]
    fn concatenated_gzip_members_share_limit_and_validate_every_member() {
        let mut encoded=gzip(b"first").to_vec();encoded.extend_from_slice(&gzip(b"second"));
        assert_eq!(decode_request_body(Bytes::from(encoded.clone()),true,11).unwrap().as_ref(),b"firstsecond");
        assert_eq!(decode_request_body(Bytes::from(encoded.clone()),true,10).unwrap_err().status,413);
        let last=encoded.len()-1;encoded[last]^=1;
        assert_eq!(decode_request_body(Bytes::from(encoded),true,100).unwrap_err().status,400);
    }

    #[test]
    fn malformed_gzip_is_rejected() {
        assert!(decode_request_body(Bytes::from_static(b"invalid"), true, 1024).is_err());
    }
}

async fn handle(node: Arc<Node>,request:Request<Body>)->Response<Body>{
    let (parts, body) = request.into_parts();
    handle_parts(node, parts, RequestBody::Stream(body)).await
}

/// Where [`handle_inner`] takes a request's body from.
enum RequestBody {
    /// A client's (or a router's) body stream: collected under the HTTP
    /// content limit and the in-flight breaker, gunzipped when encoded.
    Stream(Body),
    /// The in-process cluster front end's body, as it holds it
    /// ([`handle_in_process_ready`]): never encoded, already admitted and
    /// size-checked at its client edge.
    Ready(Bytes),
}

async fn handle_parts(node: Arc<Node>, parts: axum::http::request::Parts, body: RequestBody) -> Response<Body> {
    let control=in_flight_policy(parts.method.as_str(),parts.uri.path(),incremental_bulk(&node))==InFlight::Holds;
    let started = std::time::Instant::now();
    let r = elasticsearch_engine::memory::request_policy_future(control,handle_inner(node,parts,body)).await;
    elasticsearch_engine::search::search_phases::record_named("engine_http", vec![("total", started.elapsed().as_micros())]);
    r
}

async fn handle_inner(node: Arc<Node>, mut parts: axum::http::request::Parts, body: RequestBody) -> Response<Body> {
    let t_entry = std::time::Instant::now();
    let authenticated=security::authenticated(&node,parts.headers.get("authorization").and_then(|v|v.to_str().ok()));
    let mut metadata=match input_metadata::Builder::new(!authenticated) {
        Ok(owner)=>owner, Err(_)=>return input_metadata::refusal(parts.method==Method::HEAD),
    };
    let (path,mut params)=match (||Ok::<_,EsError>((metadata.text(parts.uri.path())?,metadata.parameters(parts.uri.query())?)))() {
        Ok(parts)=>parts, Err(_)=>return input_metadata::refusal(parts.method==Method::HEAD),
    };
    let is_head = parts.method == Method::HEAD;
    let accepts_gzip = parts.headers.get("accept-encoding").and_then(|v| v.to_str().ok()).map(|v| v.contains("gzip")).unwrap_or(false);
    let pretty = matches!(params.get("pretty").map(String::as_str), Some("") | Some("true"));
    let detailed_errors = node.node_settings.read().get("http.detailed_errors.enabled").is_none_or(|v| v != "false" && *v != Value::Bool(false));
    let mut head = ReqHead {
        detailed_errors,
        api_version: RestApiVersion::CURRENT,
        pretty,
        filter_path: match params.get("filter_path").map(|value|metadata.text(value)).transpose() { Ok(value)=>value,Err(_)=>return input_metadata::refusal(parts.method==Method::HEAD) },
        error_trace: matches!(params.get("error_trace").map(String::as_str), Some("") | Some("true")),
        is_head,
        gzip: accepts_gzip,
        content_type: "application/json".into(),
        human: false,
        wire: None,
        route_warnings: Vec::new(),
        _metadata: None,
        in_process: parts.extensions.get::<InProcessJson>().is_some(),
        typed: parts.extensions.get::<InProcessTyped>().is_some(),
    };
    let gz_in = parts.headers.get("content-encoding").and_then(|v| v.to_str().ok()).map(|v| v.contains("gzip")).unwrap_or(false);
    let body = match body {
        // Nothing to collect, decode or limit: the bytes as the front end
        // holds them (a shard request's body is its coordinator's).
        RequestBody::Ready(bytes) if !gz_in => Ok(bytes),
        body => Err(body),
    };
    // Held until the response is built (see below), on either path.
    let mut _inflight: Option<InFlightBody> = None;
    let body = match body { Ok(bytes) => bytes, Err(body) => {
    let content_limit = {
        let settings = node.node_settings.read();
        let configured = settings.get("http.max_content_length");
        match configured {
            Some(value) => {
                let text = value.as_str().map(str::to_owned).unwrap_or_else(|| format!("{value}b"));
                match elasticsearch_engine::ml::job::parse_bytes(&text, "http.max_content_length")
                    .and_then(|n| usize::try_from(n).map_err(|_| EsError::parsing("http.max_content_length exceeds address space"))) {
                    Ok(n) if n > 0 => n,
                    _ => return finish(&head, Err(EsError::parsing("invalid http.max_content_length"))),
                }
            }
            None => 100 << 20,
        }
    };
    // ES `RestController`: the body is held on the `inflight_requests`
    // breaker (overhead 2) until the response is built (see
    // [`in_flight_policy`]). A cluster's internal dispatch carries a body
    // its client edge already holds (`front::handle_http`): not charged
    // again (ES charges a body once per node).
    let policy = if parts.extensions.get::<InternalHttpDispatch>().is_some() {
        InFlight::Uncharged
    } else {
        in_flight_policy(parts.method.as_str(), &path, incremental_bulk(&node))
    };
    let authorization = parts.headers.get("authorization").and_then(|v| v.to_str().ok());
    let authenticated=security::authenticated(&node, authorization);
    let inflight = _inflight.insert(match InFlightBody::reserve(node.breakers, policy, 0, || authenticated) {
        Ok(r) => r,
        Err(e) => return finish(&head, Err(e)),
    });
    let raw = match body {
        RequestBody::Stream(body) => match collect_request_body(body,content_limit,inflight,authenticated).await {
            Ok(raw)=>raw,Err(error)=>return finish(&head,Err(error)),
        },
        RequestBody::Ready(bytes) => bytes.to_vec(),
    };
    match decode_request_body_guarded(raw, gz_in, content_limit, inflight,authenticated) {
        Ok(body) => Bytes::from(body),
        Err(error) => return finish(&head, Err(error)),
    }
    }};
    let t_body = t_entry.elapsed().as_micros();
    if let Err(error)=metadata.media(&parts.headers){return finish(&head,Err(error))}
    let api_version = match media_select(&parts.headers, params.get("format").map(String::as_str), !body.is_empty()) {
        Ok(sel) => {
            head.content_type = sel.content_type;
            head.wire = sel.wire;
            head.api_version = sel.api_version;
            sel.api_version
        }
        Err(e) => return finish(&head, Err(e)),
    };
    // YAML / CBOR / Smile bodies are decoded into JSON here, once.
    let mut source_map = None;
    let mut body = body;
    let mut headers = parts.headers;
    if !body.is_empty() {
        if let Some(ct) = headers.get("content-type") {
            let ct = match metadata.lossy(ct.as_bytes()){Ok(value)=>value,Err(error)=>return finish(&head,Err(error))};
            let fmt = xcontent_in::in_format(&ct);
            let delimited = xcontent_in::is_delimited_endpoint(&path);
            if !body_type_supported(&ct) || fmt.is_some_and(|f| delimited && !xcontent_in::supports_delimited(f)) {
                let e = serde_json::json!({"error": format!("Content-Type header [{ct}] is not supported"), "status": 406});
                return finish(&head, Ok(Resp::json(406, e)));
            }
            if let Some(f) = fmt {
                let d = match if delimited { xcontent_in::decode_stream(f, &body) } else { xcontent_in::decode(f, &body) } {
                    Ok(decoded) => decoded,
                    Err(error) => return finish(&head, Err(error)),
                };
                body = d.json;
                source_map = d.map;
                // The `Content-Type` stays: handlers that keep the request's
                // x-content type (`_ignored_source`) read it.
            }
        }
    } else if let (Some(src), Some(sct)) = (params.get("source"), params.get("source_content_type")) {
        // `?source=` with a YAML `source_content_type`.
        if xcontent_in::in_format(sct) == Some(xcontent_in::InFmt::Yaml) {
            let d = match xcontent_in::decode(xcontent_in::InFmt::Yaml, src.as_bytes()) {
                Ok(decoded) => decoded,
                Err(error) => return finish(&head, Err(error)),
            };
            if let Ok(text) = std::str::from_utf8(&d.json) {
                // Query parameters require an owned String. Admit the actual
                // copy before allocation and keep its owner with Req.
                let owner = match node.breakers.reserve_request((text.len() as u64).saturating_add(256), "decoded source parameter") {
                    Ok(owner) => owner,
                    Err(error) => return finish(&head, Err(error)),
                };
                let mut copy = String::new();
                if copy.try_reserve_exact(text.len()).is_err() {
                    return finish(&head, Err(elasticsearch_engine::memory::allocation_refused("decoded source parameter allocation refused", 0)));
                }
                copy.push_str(text);
                source_map = d.map;
                params.insert("source".into(), copy);
                params.insert("source_content_type".into(), "application/json".into());
                parts.extensions.insert(DecodedSourceOwner(std::sync::Arc::new(owner.into_result_owner())));
            }
        }
    }
    let t_media = t_entry.elapsed().as_micros();
    // Request validation prologue (ES `RestController` + `BaseRestHandler`):
    // 405 / OPTIONS before authentication, body + parameter checks after.
    let security = node.security.enabled();
    let route = {
        let uri = parts.uri.path_and_query().map(|pq| pq.as_str()).unwrap_or(path.as_str());
        match spec_validate::lookup(&parts.method, &path, uri, security, api_version) {
            spec_validate::Lookup::Early(resp) => return finish(&head, Ok(resp)),
            spec_validate::Lookup::Route(r) => Some(r),
            spec_validate::Lookup::Unmatched => None,
        }
    };
    if let Some(r) = route {
        if let Err(error)=metadata.route_warning(&parts.method,r.path){return finish(&head,Err(error))}
        head.route_warnings = deprecations::for_route(&parts.method, r.path);
    }
    let t_route = t_entry.elapsed().as_micros();
    let segs=match metadata.segments(&path){Ok(value)=>value,Err(error)=>return finish(&head,Err(error))};
    head.human = matches!(params.get("human").map(String::as_str), Some("") | Some("true")) && util::human_applies(&segs);
    // `RestController.validateErrorTrace`.
    if route.is_some() && head.error_trace && !head.detailed_errors {
        return finish(&head, Err(EsError::illegal_argument("error traces in responses are disabled.")));
    }
    let metadata_owner=input_metadata::Owner(std::sync::Arc::new(std::sync::Mutex::new(metadata.memory.into_result_owner())));
    parts.extensions.insert(metadata_owner.clone());
    // ReqHead fields drop before its final owner field.
    head._metadata=Some(metadata_owner.0);
    let req = Req { method: parts.method.clone(), path, segs, params, body, headers, access: None, extensions: parts.extensions, api_version };
    if let (Some(r), false) = (route, security) {
        if let Err(e) = spec_validate::check(&req, r) {
            return finish(&head, Err(e));
        }
    }
    let _http = if req.extensions.get::<InternalHttpDispatch>().is_some() { None } else {
        let uri = parts.uri.path_and_query().map(|pq| pq.as_str()).unwrap_or(req.path.as_str());
        Some(observability::http_request(&req.method, &req.path, uri, &req.headers, req.body.len() as u64, req.extensions.get::<axum::extract::ConnectInfo<HttpConnection>>().map(|c| &c.0)))
    };
    let diagnostic_owner=input_metadata::Owner(head._metadata.as_ref().expect("request owner").clone());
    let trace = if kibana_compat::trace_enabled() {
        let method=match diagnostic_owner.formatted(&req.method,!authenticated){Ok(value)=>value,Err(error)=>return finish(&head,Err(error))};
        let uri=match diagnostic_owner.formatted(&parts.uri,!authenticated){Ok(value)=>value,Err(error)=>return finish(&head,Err(error))};
        Some((method, uri, req.body.clone(), std::time::Instant::now()))
    } else {
        None
    };
    // Security (opt-in) wraps the router; disabled costs one atomic load.
    let pos_text = if req.body.is_empty() {
        match req.params.get("source").map(|source|input_metadata::Owner(head._metadata.as_ref().expect("request owner").clone()).bytes(source.as_bytes(),!authenticated)).transpose(){Ok(value)=>value.unwrap_or_default(),Err(error)=>return finish(&head,Err(error))}
    } else {req.body.clone()};
    // Engine hooks (semantic_text, knn `query_vector_builder`, rerankers)
    // resolve inference endpoints on this node, also when several in-process
    // nodes share the process (YAML runner).
    let inf = Some(elasticsearch_engine::inference::service_for(&node.data));
    let internal_monitor = req.extensions.get::<InternalNodeMonitor>().is_some() && req.method == Method::GET && req.seg(0) == "_nodes" && req.seg(1) == "_local";
    // Engine work that must reach every shard of a cluster (ML results,
    // datafeed sources) goes through the coordinator that sent this request.
    let coordinator = req.extensions.get::<ClusterRest>().map(|c| c.0.clone());
    // Request-breaker bytes of this request, across the threads it runs on.
    let scope_label=match diagnostic_owner.formatted(format_args!("{} {}",req.method,req.path),!authenticated){Ok(value)=>value,Err(error)=>return finish(&head,Err(error))};
    let scope = elasticsearch_engine::memory::RequestScope::with_label_owner(scope_label, node.breakers, diagnostic_owner.0.clone());
    let t_pre = t_entry.elapsed().as_micros();
    let mut routed = std::pin::pin!(async move {
        if node.security.enabled() && !internal_monitor {
            security::handle(node, req).await
        } else {
            if internal_monitor { spec_validate::validate(&req, node.security.enabled())?; }
            dispatch(node, req).await
        }
    });
    // Keep the originating owner outside the temporary poll future. Metric
    // JSON outlives dispatch and is still live while finish serializes it.
    let polling_scope = scope.clone();
    let (mut polls, mut poll_ns) = (0u64, 0u128);
    let r = std::future::poll_fn(|cx| {
        let p0 = std::time::Instant::now();
        let out = elasticsearch_engine::memory::scoped(Some(polling_scope.clone()), || {
            elasticsearch_engine::coord::scoped(coordinator.clone(), || {
                elasticsearch_engine::version::with_rest_api_version(api_version, || elasticsearch_engine::inference::scoped(inf.clone(), || routed.as_mut().poll(cx)))
            })
        });
        polls += 1;
        poll_ns += p0.elapsed().as_nanos();
        out
    })
    .await;
    elasticsearch_engine::search::search_phases::record_named("engine_polls", vec![("polls", polls as u128), ("poll_us", poll_ns / 1000)]);
    let t_dispatched = t_entry.elapsed().as_micros();
    // Body-position hints (`[line:col]`, `line`/`col`) are resolved against the raw body once, on failure.
    let r = r.map_err(|e| elasticsearch_engine::jsonpos::resolve_mapped(e, &pos_text, source_map.as_ref()));
    if let Some((m, u, b, t0)) = &trace {
        kibana_compat::trace(m, u, b, &r, *t0);
    }
    let r = match r {
        Ok(resp) if resp.status == 0 => Ok(resp),
        other => other,
    };
    // JsonResponse exposes a naked Value that callers remove from extensions.
    // Charged outputs therefore use the existing serialized-body fallback,
    // whose owner can travel with the body instead of escaping unaccounted.
    let mut head = head;
    if scope.held() > 0 { head.in_process = false; head.typed = false; }
    let resp = elasticsearch_engine::memory::scoped(Some(scope.clone()), || finish(&head, r));
    if let Some(http) = _http { http.done(&resp); }
    let t_done = t_entry.elapsed().as_micros();
    elasticsearch_engine::search::search_phases::record_named("engine_http_phases", vec![("body", t_body), ("media", t_media - t_body), ("route", t_route - t_media),
        ("prepare", t_pre - t_route), ("dispatch", t_dispatched - t_pre), ("finish", t_done - t_dispatched)]);
    let resp = if scope.held() > 0 {
        use futures::StreamExt;
        use axum::body::HttpBody;
        let length = resp.body().size_hint().exact();
        let (mut parts, body) = resp.into_parts();
        if let Some(length) = length {
            if !parts.headers.contains_key(axum::http::header::CONTENT_LENGTH) {
                parts.headers=match outgoing_headers::with_content_length(&parts.headers,length) {
                    Ok(headers)=>headers,Err(error)=>return encoding_failure(&head,&error),
                };
            }
        }
        // finish emits only data frames (including its gzip stream), never
        // trailers. Retain on full consumption AND on cancellation/drop.
        let stream = futures::stream::unfold((body.into_data_stream(), scope), |(mut body, scope)| async move {
            body.next().await.map(|chunk| (chunk, (body, scope)))
        });
        Response::from_parts(parts, Body::from_stream(stream))
    } else { resp };
    resp
}

/// A request to `node`'s REST layer from this process: what [`router`]
/// serves, without the router (it routes everything to this).
pub async fn handle_in_process(node: Arc<Node>, request: Request<Body>) -> Response<Body> {
    handle(node, request).await
}

/// [`handle_in_process`] of a request whose body the cluster front end
/// hands over as it holds it: not collected again, copied, limited or
/// decoded (a gzip-encoded one still is). Every shard request of a search
/// takes this path.
pub async fn handle_in_process_ready(node: Arc<Node>, parts: axum::http::request::Parts, body: Bytes) -> Response<Body> {
    handle_parts(node, parts, RequestBody::Ready(body)).await
}

/// The full router.
pub fn router(node: Arc<Node>) -> axum::Router {
    elasticsearch_engine::gpu::start(&node);
    static WAKE: std::sync::Once = std::sync::Once::new();
    WAKE.call_once(|| elasticsearch_engine::memory::add_wake_hook(pools::wake_idle));
    axum::Router::new().fallback(move |req: Request<Body>| {
        let node = node.clone();
        panic_guard(async move { handle(node, req).await })
    })
}

/// Run a request handler, turning a panic into an ES-shaped 500 response
/// instead of dropping the connection (release builds unwind, so a panic in
/// one request never takes the node down; see `[profile.release]`).
pub async fn panic_guard<F>(handler: F) -> Response<Body>
where
    F: std::future::Future<Output = Response<Body>>,
{
    use futures::FutureExt;
    match std::panic::AssertUnwindSafe(handler).catch_unwind().await {
        Ok(response) => response,
        Err(payload) => {
            let reason = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "request handler panicked".to_string());
            tracing::error!(%reason, "request handler panicked");
            panic_response(&reason)
        }
    }
}

/// The 500 body for a panicked request: ES's shape for an unexpected
/// server-side exception.
pub fn panic_response(reason: &str) -> Response<Body> {
    let reason = format!("internal error: {reason}");
    let body = serde_json::json!({
        "error": {
            "root_cause": [{"type": "exception", "reason": reason}],
            "type": "exception",
            "reason": reason,
        },
        "status": 500,
    });
    let mut response = Response::new(Body::from(body.to_string()));
    *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
    response.headers_mut().insert("content-type", axum::http::HeaderValue::from_static("application/json"));
    response.headers_mut().insert("x-elastic-product", axum::http::HeaderValue::from_static("Elasticsearch"));
    response
}

/// Serve on `addr` until the process ends.
pub async fn serve(node: Arc<Node>, addr: SocketAddr) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    serve_listener(node, listener).await
}

/// Serve on an already bound listener (tests bind port 0).
fn http_addrs() -> &'static std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, String>> {
    static A: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, String>>> = std::sync::OnceLock::new();
    A.get_or_init(Default::default)
}

/// The HTTP address `node` is bound to (`ES_HTTP_ADDR`, else the address of
/// the listener it serves on, else `127.0.0.1:9200`).
pub(crate) fn http_addr(node: &Node) -> String {
    std::env::var("ES_HTTP_ADDR").ok().or_else(|| http_addrs().lock().ok().and_then(|m| m.get(&node.data).cloned())).unwrap_or_else(|| "127.0.0.1:9200".into())
}

pub async fn serve_listener(node: Arc<Node>, listener: tokio::net::TcpListener) -> std::io::Result<()> {
    serve_listener_with_tls(node, listener, None).await
}

/// [`serve_listener`], over HTTPS when `tls` is set (the server config of
/// `xpack.security.http.ssl.*`, see `elasticsearch_ssl::http_server`).
pub async fn serve_listener_with_tls(node: Arc<Node>, listener: tokio::net::TcpListener, tls: Option<Arc<elasticsearch_ssl::rustls::ServerConfig>>) -> std::io::Result<()> {
    if let Ok(a) = listener.local_addr() {
        if let Ok(mut m) = http_addrs().lock() {
            m.insert(node.data.clone(), a.to_string());
        }
    }
    // Lifecycle executor (ILM / data stream lifecycle / SLM schedules).
    elasticsearch_engine::ilm::start(&node);
    elasticsearch_engine::watcher::start(&node);
    // Persisted shard follow tasks / auto-follow patterns (CCR).
    elasticsearch_engine::ccr::start(&node);
    // Self-generated license (basic, or `xpack.license.self_generated.type`).
    elasticsearch_engine::license::start(&node);
    // Built-in template registries (stack, apm-data, otel-data).
    elasticsearch_engine::registry::ensure(&node);
    if tls.is_some() {
        node.security.cfg.write().http_ssl = true;
    }
    let app = router(node).into_make_service_with_connect_info::<HttpConnection>();
    match tls {
        Some(config) => {
            let config = Arc::new(elasticsearch_ssl::Reloadable::fixed(elasticsearch_ssl::HTTP_SSL.trim_end_matches('.'), config));
            axum::serve(TlsListener::new(listener, config)?, app).await
        }
        None => axum::serve(listener, app).await,
    }
}

pub(crate) fn not_found_handler(req: &Req) -> R {
    Ok(no_handler(req))
}

/// 405 for a known path with the wrong method.
pub(crate) fn method_not_allowed(req: &Req, allowed: &[&str]) -> R {
    Ok(Resp {
        status: 405,
        body: RespBody::Json(serde_json::json!({
            "error": format!("Incorrect HTTP method for uri [{}] and method [{}], allowed: [{}]", req.path, req.method, allowed.join(", ")),
            "status": 405
        })),
        headers: vec![("Allow", allowed.join(","))],
    })
}

#[cfg(test)]
mod media_tests {
    #[test]
    fn non_detailed_errors() {
        let e = EsError::new(404, "index_not_found_exception", "no such index [x]");
        let (st, body, h) = super::non_detailed_error(&e, RestApiVersion::V9);
        let RespBody::Json(v) = body else { panic!() };
        assert_eq!((st, v, h.len()), (404, serde_json::json!({"error": {"type": "index_not_found_exception", "reason": "no such index [x]"}, "status": 404}), 0));
        let (_, body, h) = super::non_detailed_error(&e, RestApiVersion::V8);
        let RespBody::Json(v) = body else { panic!() };
        assert_eq!(v["error"], "IndexNotFoundException[no such index [x]]");
        assert_eq!(h.len(), 1);
        let (_, body, _) = super::non_detailed_error(&EsError::new(400, "parse_exception", "p"), RestApiVersion::V8);
        let RespBody::Json(v) = body else { panic!() };
        assert_eq!(v["error"], "ElasticsearchParseException[p]");
    }

    use super::*;

    fn sel(accept: Option<&str>, ctype: Option<&str>, format: Option<&str>, body: bool) -> Result<MediaSel, EsError> {
        let mut h = HeaderMap::new();
        if let Some(a) = accept {
            h.insert("accept", a.parse().unwrap());
        }
        if let Some(c) = ctype {
            h.insert("content-type", c.parse().unwrap());
        }
        media_select(&h, format, body)
    }

    #[test]
    fn rest_api_version_like_es_9_5_5() {
        let vnd = |n: u8| format!("application/vnd.elasticsearch+json;compatible-with={n}");
        assert_eq!(sel(None, None, None, false).unwrap().api_version, RestApiVersion::V9);
        let s = sel(Some(&vnd(8)), None, None, false).unwrap();
        assert_eq!(s.api_version, RestApiVersion::V8);
        assert_eq!(s.content_type, "application/vnd.elasticsearch+json;compatible-with=8");
        assert_eq!(sel(Some(&vnd(9)), None, None, false).unwrap().api_version, RestApiVersion::V9);
        assert_eq!(sel(Some(&vnd(8)), Some(&vnd(8)), None, true).unwrap().api_version, RestApiVersion::V8);
        // 7 is rejected (ES 9 knows only V_9 and V_8).
        let e = sel(Some(&vnd(7)), None, None, false).unwrap_err();
        assert_eq!(e.kind, "media_type_header_exception");
        let v = e.to_json_with_trace(false);
        assert_eq!(v["error"]["caused_by"]["reason"], "Accept version must be either version 9 or 8, but found 7. Accept=application/vnd.elasticsearch+json;compatible-with=7");
        // Versioned on one header only, with a body.
        let e = sel(Some(&vnd(8)), Some("application/json"), None, true).unwrap_err();
        assert!(e.to_json_with_trace(false)["error"]["caused_by"]["reason"].as_str().unwrap().contains("the compatible versions must match"));
        // ... without a body the Content-Type version is not checked.
        assert_eq!(sel(Some(&vnd(8)), Some("application/json"), None, false).unwrap().api_version, RestApiVersion::V8);
        // Media ranges are ignored; malformed types are rejected.
        assert_eq!(sel(Some("application/vnd.elasticsearch+json; compatible-with=8,text/plain"), None, None, false).unwrap().api_version, RestApiVersion::V9);
        assert_eq!(sel(Some("garbage"), None, None, false).unwrap_err().reason, "Invalid media-type value on headers [Accept]");
    }

    #[test]
    fn response_content_type_like_es() {
        assert_eq!(sel(None, None, None, false).unwrap().content_type, "application/json");
        assert_eq!(sel(Some("application/json; charset=UTF-8"), None, None, false).unwrap().content_type, "application/json;charset=utf-8");
        assert_eq!(sel(Some("text/plain"), None, None, false).unwrap().content_type, "application/json");
        assert_eq!(sel(Some("application/vnd.elasticsearch+json"), None, None, false).unwrap().content_type, "application/vnd.elasticsearch+json");
        assert_eq!(sel(Some("application/vnd.elasticsearch+x-ndjson;compatible-with=8"), None, None, false).unwrap().content_type, "application/vnd.elasticsearch+json;compatible-with=8");
        let y = sel(Some("application/vnd.elasticsearch+json;compatible-with=8"), None, Some("yaml"), false).unwrap();
        assert_eq!((y.content_type.as_str(), y.wire.is_some()), ("application/yaml;compatible-with=8", true));
        assert_eq!(sel(None, Some("application/vnd.elasticsearch+json;compatible-with=8"), None, false).unwrap().content_type, "application/vnd.elasticsearch+json");
    }
}

#[cfg(test)]
mod malformed_percent_regression {
    #[test]
    fn malformed_percent_before_unicode_preserves_literal_text() {
        for input in ["%λ", "%€", "a%😀b", "%aλ", "%", "%a"] {
            assert_eq!(super::decode_component(input), input);
            assert_eq!(super::decode_path_segment(input), input);
        }
        assert_eq!(super::decode_component("%2b+a"), "+ a");
        assert_eq!(super::decode_path_segment("%2B+a"), "++a");
    }
}

#[cfg(test)]
mod authentication_output_tests {
    use super::*;
    #[tokio::test]
    async fn response_refusal_preserves_authentication_and_head_semantics() {
        use elasticsearch_engine::memory::{self, Breakers, Kind, RequestScope};
        for parent in [false, true] {
            let breakers = Breakers::leak_for_tests(1024);
            breakers.apply_settings(&|key| match key {
                "indices.breaker.total.use_real_memory" => Some("false".into()),
                "indices.breaker.total.limit" => Some(if parent { "1b" } else { "-1" }.into()),
                "indices.breaker.request.limit" => Some(if parent { "-1" } else { "1b" }.into()),
                _ => None,
            });
            let held = breakers.reserve_without_breaking(Kind::Request, 1024);
            let scope = RequestScope::new("authentication output refusal", breakers);
            for is_head in [false, true] {
                let head = ReqHead { detailed_errors: true, api_version: RestApiVersion::CURRENT,
                    pretty: false, filter_path: None, error_trace: false, is_head, gzip: false,
                    content_type: "application/json".into(), human: false, wire: None,
                    in_process: false, typed: false, route_warnings: Vec::new(), _metadata: None };
                let response = memory::scoped(Some(scope.clone()), || {
                    assert!(breakers.reserve_request(8, "forced refusal").is_err());
                    finish(&head, Err(EsError::new(401, "security_exception", "missing credentials")))
                });
                assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
                assert!(response.headers().contains_key("www-authenticate"));
                let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
                if is_head { assert!(body.is_empty()); }
                else { assert_eq!(serde_json::from_slice::<Value>(&body).unwrap()["status"], 401); }
            }
            drop(scope); drop(held);
            assert_eq!(breakers.request.reserved(), 0);
        }
    }
}
