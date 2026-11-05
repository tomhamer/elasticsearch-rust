//! Document APIs: index / create / get / _source / delete / update / mget / bulk.

use crate::util::{cond_from_meta, handle_refresh, handle_refresh_doc, write_cond, write_json};
use crate::{blocking, method_not_allowed, Req, Resp, RespBody, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::index::{IndexService, WriteReq};
use elasticsearch_engine::node::Node;
use elasticsearch_engine::search::{doc_visible, filter_source, parse_source_filter, SourceFilter};
use elasticsearch_engine::security::roles::IndexAccess;
use elasticsearch_engine::shard::{GetResult, WriteCond};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

#[path = "docs_stored.rs"]
mod docs_stored;

pub async fn handle(node: Arc<Node>, req: Req, index: Option<String>) -> R {
    let segs: Vec<String> = req.segs.clone();
    let off = if index.is_some() { 1 } else { 0 };
    let api = segs.get(off).cloned().unwrap_or_default();
    let id = segs.get(off + 1).cloned();
    let m = req.method.clone();
    match (api.as_str(), id, &m) {
        ("_bulk", None, &Method::POST) | ("_bulk", None, &Method::PUT) => {
            // The translog sync request durability needs runs after the
            // write thread is released (`sync_deferred`); the response
            // still waits for it.
            let deferred: DeferredSyncs = Default::default();
            let syncs = deferred.clone();
            let r = blocking(move || bulk(&node, &req, index.as_deref(), Some(&syncs))).await;
            let pending = std::mem::take(&mut *deferred.lock().unwrap_or_else(|e| e.into_inner()));
            if r.is_err() || pending.is_empty() { return r; }
            sync_deferred(pending).await?;
            r
        }
        ("_bulk", _, _) => method_not_allowed(&req, &["POST", "PUT"]),
        ("_mget", None, &Method::GET) | ("_mget", None, &Method::POST) => blocking(move || mget(&node, &req, index.as_deref())).await,
        ("_doc", None, &Method::POST) => blocking(move || index_doc(&node, &req, &index.unwrap(), None, false)).await,
        ("_doc", Some(id), &Method::PUT) | ("_doc", Some(id), &Method::POST) => blocking(move || index_doc(&node, &req, &index.unwrap(), Some(id), false)).await,
        ("_create", Some(id), &Method::PUT) | ("_create", Some(id), &Method::POST) => blocking(move || index_doc(&node, &req, &index.unwrap(), Some(id), true)).await,
        ("_doc", Some(id), &Method::GET) | ("_doc", Some(id), &Method::HEAD) => blocking(move || get_doc(&node, &req, &index.unwrap(), &id, false)).await,
        ("_source", Some(id), &Method::GET) | ("_source", Some(id), &Method::HEAD) => blocking(move || get_doc(&node, &req, &index.unwrap(), &id, true)).await,
        ("_doc", Some(id), &Method::DELETE) => blocking(move || delete_doc(&node, &req, &index.unwrap(), &id)).await,
        ("_update", Some(id), &Method::POST) => blocking(move || update_doc(&node, &req, &index.unwrap(), &id)).await,
        ("_doc", Some(_), _) => method_not_allowed(&req, &["GET", "HEAD", "PUT", "POST", "DELETE"]),
        _ => crate::not_found_handler(&req),
    }
}

/// `index.translog.durability`: `request` (default) fsyncs the translog
/// before a write (single doc or a whole `_bulk`) is acknowledged.
fn sync_mode(svc: &IndexService) -> bool {
    // Request durability must remain the ES9 default. Async acknowledgement
    // is an explicit index setting, not a substitute for batching durable
    // writes. The performance comparisons set request durability on both
    // engines, so restoring this default preserves their acknowledgement gate.
    svc.setting("index.translog.durability").map(|d| !d.eq_ignore_ascii_case("async")).unwrap_or(true)
}

/// `require_alias`: the target must be an alias.
fn require_alias(node: &Node, idx: &str, flag: bool) -> Result<(), EsError> {
    if flag && !node.all_indices().iter().any(|s| s.aliases().contains_key(idx)) {
        return Err(EsError::new(404, "index_not_found_exception", format!("no such index [{idx}] and [require_alias] request flag is [true] and [{idx}] is not an alias"))
            .with("resource.type", "index_expression")
            .with("resource.id", idx)
            .with("index_uuid", "_na_")
            .with("index", idx));
    }
    Ok(())
}

/// `set_security_user` needs the authenticated user on the thread that runs the pipelines.
/// Enter the request's dynamic-mapping authorizer (security) for the writes
/// this handler performs.
fn mapping_authz(req: &Req) -> elasticsearch_engine::index::MappingAuthzGuard {
    elasticsearch_engine::index::enter_mapping_authz(req.extensions.get::<crate::security::authz::MappingAuthz>().map(|m| m.0.clone()))
}

fn ingest_user(req: &Req) -> Option<elasticsearch_engine::ingest::IngestContextGuard> {
    req.access.as_deref().map(|a| elasticsearch_engine::ingest::IngestContext::enter(elasticsearch_engine::ingest::IngestContext { user: Some(a.user().clone()) }))
}

/// `_ignored_source` objects keep the indexing request's XContent type
/// (`XContentDataHelper`: a YAML / CBOR / Smile body stores them as such).
fn ignored_source_xcontent(req: &Req) -> Option<elasticsearch_engine::synthetic::XContentGuard> {
    use crate::wire::{encode, Wire};
    // The cluster edge forwards decoded bodies as JSON and names the original type.
    let ct = req.headers.get(crate::xcontent_in::ORIGINAL_CONTENT_TYPE).or_else(|| req.headers.get("content-type"))?.to_str().ok()?.to_ascii_lowercase();
    let mime = ct.split(';').next().unwrap_or("").trim();
    let kind = mime.strip_prefix("application/vnd.elasticsearch+").or_else(|| mime.strip_prefix("application/"))?;
    Some(match kind {
        "yaml" => elasticsearch_engine::synthetic::ignored_source_xcontent(b'y', |v| encode(Wire::Yaml, v)),
        "cbor" => elasticsearch_engine::synthetic::ignored_source_xcontent(b'c', |v| encode(Wire::Cbor, v)),
        "smile" => elasticsearch_engine::synthetic::ignored_source_xcontent(b's', |v| encode(Wire::Smile, v)),
        _ => return None,
    })
}

fn index_doc(node: &Node, req: &Req, index: &str, id: Option<String>, create: bool) -> R {
    let _xcontent = ignored_source_xcontent(req);
    let _ingest_user = ingest_user(req);
    let _mapping_authz = mapping_authz(req);
    require_alias(node, index, req.param("require_alias") == Some("true"))?;
    node.require_data_stream(index, req.param("require_data_stream") == Some("true"))?;
    let mut cond = write_cond(req)?;
    if create {
        cond.create = true;
    }
    let src = req.body_str()?.to_string();
    if src.trim().is_empty() {
        return Err(EsError::action_validation(&["source is missing", "content type is missing"]).with_status(400));
    }
    // An alias's `index_routing` routes the document.
    let routing = node.index_routing_for(index, req.param("routing"))?;
    let original_index = index.to_string();
    let (orig_id, orig_routing) = (id.clone(), routing.clone());
    // Ingest pipelines (`?pipeline=`, index default/final pipelines).
    let ingest = match req.extensions.get::<crate::PreparedBulkIngestItems>().and_then(|items| items.0.get(&0)) {
        Some(crate::PreparedBulkIngest::Write { routing, dynamic_templates }) => Ok(crate::ingest::Prepared::Write {
            index: index.to_string(), id, routing: routing.clone(), source: src, dynamic_templates: dynamic_templates.clone(),
        }),
        _ => crate::ingest::prepare_ext(node, index, id, routing, src, req.param("pipeline")),
    };
    let (index, id, routing, src, ingest_dt) = match ingest {
        Ok(crate::ingest::Prepared::Write { index, id, routing, source, dynamic_templates }) => (index, id, routing, source, dynamic_templates),
        Ok(crate::ingest::Prepared::Dropped { index, id }) => return Ok(Resp::ok(Value::Object(crate::ingest::dropped_json(&index, id.as_deref())))),
        // An ingest failure goes to the failure store of the index it was headed for.
        Err(b) => {
            let (f, original_src) = *b;
            return redirect_failure(req, true, node.failure_store_target(&f.index), f.error, |ds, e| {
                store_failure(node, req, ds, &original_index, orig_id.as_deref(), orig_routing.as_deref(), &original_src, e, Some(&f.context), None)
            });
        }
    };
    let index = index.as_str();
    let svc = node.write_index(index, true)?;
    if req.extensions.get::<crate::InternalDataStreamBackingWrite>().is_none() {
        node.check_backing_write(index, cond.create, cond.if_seq_no.is_some() || cond.if_primary_term.is_some())?;
    }
    // Without an `_id` (and no explicit `op_type=index`) the REST layer creates.
    let mut cond = cond;
    if id.is_none() && req.param("op_type").is_none() {
        cond.create = true;
    }
    // Shard-level failures of data stream writes may go to the failure store.
    let fs_index = failure_store_index(node, req, index);
    let fs = node.failure_store_target(&fs_index);
    let keep = (!matches!(fs, elasticsearch_engine::datastream::FailureStoreTarget::None)).then(|| (src.clone(), id.clone(), routing.clone()));
    let doc_routing = routing.clone();
    let mut svc = svc;
    let res = node.check_data_stream_write_ts(index, cond.create || id.is_none(), Some(&src)).and_then(|ts| {
        // A time series data stream writes into the backing index covering
        // the document's `@timestamp` (read by the check above).
        if let Some(t) = node.time_series_write_index_at(index, ts.as_ref()) {
            svc = t;
        }
        let svc = &svc;
        let write = || svc.write(vec![WriteReq::Index { id, source: src, routing, cond }], sync_mode(svc)).remove(0);
        // `_dynamic_templates` set by an ingest pipeline apply to this write.
        match ingest_dt {
            Some(dt) => elasticsearch_engine::document::with_dynamic_templates(dt, write),
            None => write(),
        }
    });
    let r = match (res, keep) {
        (Ok(r), _) => r,
        (Err(e), Some((src, id, routing))) if failure_store_eligible(&e) => {
            let doc_id = id.or_else(|| failed_doc_id(&e));
            // Coordinating-level routing errors carry the header on the root cause too.
            let wrapped = e.kind != "timestamp_error";
            return redirect_failure(req, wrapped, fs, e, |ds, e| store_failure(node, req, ds, &fs_index, doc_id.as_deref(), routing.as_deref(), &src, e, None, None));
        }
        (Err(e), _) => return Err(e),
    };
    let forced = handle_refresh_doc(req.param("refresh"), &svc, Some((&r.id, doc_routing.as_deref())))?;
    let status = if r.result == "created" { 201 } else { 200 };
    Ok(Resp::json(status, Value::Object(write_json(&svc, &r, forced))))
}

/// The (possibly auto-generated) id of a document a shard failed to index.
fn failed_doc_id(e: &EsError) -> Option<String> {
    if let Some(id) = e.extra.iter().find(|(k, _)| k == "_id").and_then(|(_, v)| v.as_str()) {
        return Some(id.to_string());
    }
    let rest = e.reason.split("in document with id '").nth(1)?;
    rest.split('\'').next().map(str::to_string)
}

/// Whether a failed write may go to a failure store (not version
/// conflicts, nor the data stream `op_type` check).
/// The index whose failure store a write redirects to: a coordinator-routed
/// data stream write arrives addressed to its backing index, which stands
/// for the stream it belongs to.
fn failure_store_index(node: &Node, req: &Req, index: &str) -> String {
    if req.extensions.get::<crate::InternalDataStreamBackingWrite>().is_some() {
        if let Some(stream) = node.data_stream_of_index(index) {
            return stream;
        }
    }
    index.to_string()
}

fn failure_store_eligible(e: &EsError) -> bool {
    elasticsearch_engine::datastream::failure_store_eligible(e) && !e.reason.starts_with("only write ops with an op_type of create")
}

/// Single-document response for a failed write: redirected to the failure
/// store (`failure_store: used`), or the error with `failure_store: not_enabled`.
enum StoredFailure {
    Local(Arc<IndexService>, elasticsearch_engine::shard::WriteResult),
    Cluster(Value),
}

fn store_failure(node: &Node, req: &Req, stream: &str, index: &str, id: Option<&str>, routing: Option<&str>, source: &str,
    error: &EsError, context: Option<&elasticsearch_engine::datastream::FailureContext>, position: Option<usize>) -> Result<StoredFailure, EsError> {
    if let Some(writer) = req.extensions.get::<crate::FailureStoreWriter>() {
        let source: Value = serde_json::from_str(source).map_err(|_| error.clone())?;
        if !source.is_object() { return Err(error.clone()); }
        let document = elasticsearch_engine::datastream::failure_document(index, id, routing, &source, error, context);
        return (writer.0)(stream, document, position).map(StoredFailure::Cluster);
    }
    node.store_failure(stream, index, id, routing, source, error, context).map(|(service, result)| StoredFailure::Local(service, result))
}

fn redirect_failure(
    req: &Req,
    wrapped: bool,
    target: elasticsearch_engine::datastream::FailureStoreTarget,
    e: EsError,
    store: impl FnOnce(&str, &EsError) -> Result<StoredFailure, EsError>,
) -> R {
    use elasticsearch_engine::datastream::FailureStoreTarget;
    match target {
        FailureStoreTarget::Enabled(ds) => {
            let (svc, r) = match store(&ds, &e)? {
                StoredFailure::Local(service, result) => (service, result),
                StoredFailure::Cluster(mut body) => {
                    body["failure_store"] = json!("used");
                    let mut response = Resp::json(201, body);
                    response.headers.push(("x-rustic-internal-failure-store-ack", "true".into()));
                    return Ok(response);
                }
            };
            let forced = handle_refresh(req.param("refresh"), &svc)?;
            let mut m = write_json(&svc, &r, forced);
            m.insert("failure_store".into(), json!("used"));
            Ok(Resp::json(201, Value::Object(m)))
        }
        // Ingest failures are wrapped (the root cause lacks the header);
        // others carry it on the root cause too.
        FailureStoreTarget::Disabled if wrapped => {
            let mut body = e.to_json();
            body["error"]["failure_store"] = json!("not_enabled");
            Ok(Resp::json(e.status, body))
        }
        FailureStoreTarget::Disabled => Err(e.with("failure_store", "not_enabled")),
        FailureStoreTarget::None => Err(e),
    }
}

trait WithStatus {
    fn with_status(self, s: u16) -> Self;
}
impl WithStatus for EsError {
    fn with_status(mut self, s: u16) -> Self {
        self.status = s;
        self
    }
}

pub fn source_filter_from_params(req_params: &HashMap<String, String>) -> Option<SourceFilter> {
    let inc = req_params.get("_source_includes").or(req_params.get("_source_include"));
    let exc = req_params.get("_source_excludes").or(req_params.get("_source_exclude"));
    let list = |s: &String| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect::<Vec<_>>();
    if inc.is_some() || exc.is_some() {
        return Some(SourceFilter::Filter { includes: inc.map(list).unwrap_or_default(), excludes: exc.map(list).unwrap_or_default() });
    }
    match req_params.get("_source").map(String::as_str) {
        None => None,
        Some("true") | Some("") => Some(SourceFilter::All),
        Some("false") => Some(SourceFilter::None),
        Some(s) => Some(SourceFilter::Filter { includes: list(&s.to_string()), excludes: vec![] }),
    }
}

/// Render a get result (found) as a JSON string with the raw source spliced.
pub fn get_json(svc: &IndexService, id: &str, g: &GetResult, filter: &SourceFilter) -> Result<String, EsError> {
    let mut s = format!("{{\"_index\":{},\"_id\":{},\"_version\":{}", serde_json::to_string(&svc.name)?, serde_json::to_string(id)?, g.version);
    if !svc.seq_no_disabled() {
        s.push_str(&format!(",\"_seq_no\":{},\"_primary_term\":{}", g.seq_no, g.primary_term));
    }
    if let Some(r) = g.routing.as_ref().filter(|_| !svc.meta().mappings.routing_doc_values()) {
        s.push_str(&format!(",\"_routing\":{}", serde_json::to_string(r)?));
    }
    s.push_str(",\"found\":true");
    if let Some(src) = g.source.as_deref().filter(|x| !x.is_empty()) {
        match filter {
            SourceFilter::All => {
                s.push_str(",\"_source\":");
                s.push_str(src);
            }
            SourceFilter::None => {}
            SourceFilter::Filter { includes, excludes } => {
                s.push_str(",\"_source\":");
                match serde_json::from_str::<Value>(src) {
                    Ok(v) => s.push_str(&serde_json::to_string(&elasticsearch_engine::vector_source::filter_loaded(&v, includes, excludes, &svc.lookup()))?),
                    Err(_) => s.push_str(&elasticsearch_engine::source_json::filter_raw(src, includes, excludes)?),
                }
            }
        }
    }
    s.push('}');
    Ok(s)
}

/// The DLS/FLS of the caller on `svc` for single-document reads.
fn access_of(req: &Req, svc: &Arc<IndexService>) -> Result<Option<Arc<IndexAccess>>, EsError> {
    match &req.access {
        Some(a) => a.doc_access(&svc.name, "indices:data/read/get"),
        None => Ok(None),
    }
}

fn not_found_json(index: &str, id: &str) -> Value {
    json!({"_index": index, "_id": id, "found": false})
}

/// Document- and field-level security for a single-document read: a document hidden by
/// DLS is "missing"; FLS removes ungranted fields from the `_source` before anything else
/// (user `_source` filters, `stored_fields`) sees it.
fn secure_get(svc: &Arc<IndexService>, id: &str, access: &Option<Arc<IndexAccess>>, found: Option<GetResult>) -> Result<Option<GetResult>, EsError> {
    let Some(acc) = access else { return Ok(found) };
    let Some(mut g) = found else { return Ok(None) };
    if !doc_visible(svc, id, acc, true)? {
        return Ok(None);
    }
    if let Some(fa) = &acc.fields {
        // Unparseable source is withheld rather than returned unfiltered.
        g.source = g.source.as_deref().map(|raw| serde_json::from_str::<Value>(raw).map(|v| fa.filter_source(&v).to_string()).or_else(|_|elasticsearch_engine::source_json::Source::parse(raw).map(|s|s.retain_leaves(&|p|fa.allows(p)).text())).unwrap_or_default());
    }
    Ok(Some(g))
}

/// ES 9 `exclude_source_vectors`: a get leaves the vector fields out of
/// `_source` unless `_source_exclude_vectors=false` (or the index keeps
/// them), keeping the ones `_source_includes` names.
fn exclude_vectors(req: &Req, svc: &IndexService, found: Option<GetResult>, filter: &SourceFilter) -> Option<GetResult> {
    let explicit = req.param("_source_exclude_vectors").map(|v| v == "true" || v.is_empty());
    if req.param("force_synthetic_source") == Some("true") {
        return found;
    }
    found.map(|mut g| {
        if let Some(raw) = g.source.as_deref() {
            let includes: &[String] = match filter {
                SourceFilter::Filter { includes, .. } => includes,
                _ => &[],
            };
            let index_excludes = elasticsearch_engine::vector_source::excludes_source_vectors(|k| svc.setting(k));
            if let Some(new) = elasticsearch_engine::vector_source::for_read(raw, &svc.lookup(), index_excludes, explicit, includes) {
                g.source = Some(new);
            }
        }
        g
    })
}

/// `force_synthetic_source=true`: serve the `_source` of a stored-source
/// index through the synthetic loader.
fn force_synthetic(req: &Req, svc: &Arc<IndexService>, found: Option<GetResult>) -> Option<GetResult> {
    if req.param("force_synthetic_source") != Some("true") {
        return found;
    }
    found.map(|mut g| {
        if let Some(s) = g.source.as_deref().filter(|s| !s.is_empty()) {
            if let Some(f) = elasticsearch_engine::synthetic::force(svc, s) {
                g.source = Some(f);
            }
        }
        g
    })
}

/// ES `IndexNameExpressionResolver` deprecation for a request without
/// system-index access that resolves to a system index (here: the security
/// index behind the `.security` alias).
fn system_index_warning(name: &str) -> Option<(&'static str, String)> {
    name.starts_with(".security-").then(|| {
        let msg = format!("this request accesses system indices: [{name}], but in a future major version, direct access to system indices will be prevented by default");
        ("Warning", format!("299 Elasticsearch-{}-{} \"{msg}\"", elasticsearch_engine::node::ES_VERSION, crate::cluster::BUILD_HASH))
    })
}

fn get_doc(node: &Node, req: &Req, index: &str, id: &str, source_only: bool) -> R {
    let svc = node.write_index(index, false)?;
    let mut r = get_doc_in(node, req, svc.clone(), index, id, source_only)?;
    if let Some(w) = system_index_warning(&svc.name) {
        r.headers.push(w);
    }
    Ok(r)
}

fn get_doc_in(node: &Node, req: &Req, svc: Arc<IndexService>, index: &str, id: &str, source_only: bool) -> R {
    if req.flag("refresh")? {
        svc.refresh()?;
    }
    let filter = source_filter_from_params(&req.params).unwrap_or(SourceFilter::All);
    let realtime = req.bool_param("realtime")?.unwrap_or(true);
    let access = match &req.access {
        Some(a) => a.doc_access(&svc.name, "indices:data/read/get")?,
        None => None,
    };
    let routing = node.index_routing_for(index, req.param("routing"))?;
    let found = force_synthetic(req, &svc, secure_get(&svc, id, &access, svc.get_routed(id, routing.as_deref(), realtime)?)?);
    let found = exclude_vectors(req, &svc, found, &filter);
    match found {
        None => {
            if source_only {
                return Err(EsError::resource_not_found(format!("Document not found [{}]/[{id}]", svc.name)));
            }
            Ok(Resp::json(404, not_found_json(&svc.name, id)))
        }
        Some(g) => {
            if let Some(v) = req.param("version") {
                if v.parse::<u64>().ok() != Some(g.version) {
                    return Err(EsError::version_conflict(&svc.name, &svc.uuid, id, format!("version conflict, current version [{}] is different than the one provided [{v}]", g.version)));
                }
            }
            if source_only {
                let src = g.source.clone().unwrap_or_default();
                if src.is_empty() {
                    return Err(EsError::resource_not_found(format!("Source not found [{}]/[{id}]", svc.name)));
                }
                return Ok(match filter {
                    SourceFilter::Filter { includes, excludes } => match serde_json::from_str::<Value>(&src) {
                        Ok(v) => Resp::ok(elasticsearch_engine::vector_source::filter_loaded(&v, &includes, &excludes, &svc.lookup())),
                        Err(_) => Resp::raw(200, elasticsearch_engine::source_json::filter_raw(&src, &includes, &excludes)?),
                    },
                    _ => Resp::raw(200, src),
                });
            }
            if let Some(sf) = req.param("stored_fields") {
                let mut sf = docs_stored::list(&json!(sf));
                // An explicit `_source` request alongside `stored_fields` still returns the (filtered) source.
                let explicit = source_filter_from_params(&req.params).filter(|f| !matches!(f, SourceFilter::None));
                if explicit.is_some() && !sf.iter().any(|s| s == "_source") {
                    sf.push("_source".to_string());
                }
                return Ok(Resp::raw(200, docs_stored::apply(&svc, &get_json(&svc, id, &g, explicit.as_ref().unwrap_or(&SourceFilter::All))?, &sf, &g)?));
            }
            let out = get_json(&svc, id, &g, &filter)?;
            let ev = match req.param("_source_exclude_vectors") {
                None => None,
                Some(v) => Some(v.is_empty() || v == "true"),
            };
            if elasticsearch_engine::inference::semantic::rehydrate_inference_fields(&filter, ev) {
                if let Some(inf) = inference_fields_of(&svc, id, routing.as_deref())? {
                    let mut v: Value = serde_json::from_str(&out)?;
                    if let Some(Value::Object(src)) = v.get_mut("_source") {
                        src.insert("_inference_fields".into(), inf);
                    }
                    return Ok(Resp::ok(v));
                }
            }
            Ok(Resp::raw(200, out))
        }
    }
}

/// The `_inference_fields` of an indexed document (`ShardGetService` adds the
/// inference metadata stored field when it is to be rehydrated).
fn inference_fields_of(svc: &Arc<IndexService>, id: &str, routing: Option<&str>) -> Result<Option<Value>, EsError> {
    if !svc.lookup().fields.values().any(|f| elasticsearch_engine::inference::semantic::is_inference_field(f)) {
        return Ok(None);
    }
    let body = json!({"query": {"ids": {"values": [id]}}, "size": 1, "_source": {"includes": ["_inference_fields"]}});
    let mut params = HashMap::new();
    if let Some(r) = routing {
        params.insert("routing".to_string(), r.to_string());
    }
    let sreq = elasticsearch_engine::search::parse_request(&body, &params)?;
    let opts = elasticsearch_engine::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: 65_536, scroll_id: None, pit_id: None };
    let out = elasticsearch_engine::search::execute(&[elasticsearch_engine::search::Target::new(svc.clone())], &sreq, &opts)?;
    let v: Value = serde_json::from_str(&out.json)?;
    Ok(v["hits"]["hits"][0]["_source"].get("_inference_fields").cloned())
}

fn delete_doc(node: &Node, req: &Req, index: &str, id: &str) -> R {
    let svc = node.write_index(index, false)?;
    node.check_data_stream_write(index, false, None)?;
    let cond = write_cond(req)?;
    let routing = node.index_routing_for(index, req.param("routing"))?;
    let r = svc.write(vec![WriteReq::Delete { id: id.to_string(), routing: routing.clone(), cond }], sync_mode(&svc)).remove(0)?;
    let forced = handle_refresh_doc(req.param("refresh"), &svc, Some((id, routing.as_deref())))?;
    let status = if r.result == "not_found" { 404 } else { 200 };
    Ok(Resp::json(status, Value::Object(write_json(&svc, &r, forced))))
}

fn update_doc(node: &Node, req: &Req, index: &str, id: &str) -> R {
    let _mapping_authz = mapping_authz(req);
    require_alias(node, index, req.param("require_alias") == Some("true"))?;
    let svc = node.write_index(index, true)?;
    node.check_data_stream_write(index, false, None)?;
    let (body, raw_source)=match req.json(){Ok(v)=>(v,None),Err(e)=>{
        let raw=std::str::from_utf8(&req.body).map_err(|_|e)?;
        elasticsearch_engine::source_json::update_request(raw)?
    }};
    let mut cond = WriteCond::default();
    let c = write_cond(req)?;
    cond.if_seq_no = c.if_seq_no;
    cond.if_primary_term = c.if_primary_term;
    if let Some(o) = body.as_object() {
        const KNOWN: &[&str] = &["doc", "upsert", "doc_as_upsert", "script", "scripted_upsert", "detect_noop", "_source", "if_seq_no", "if_primary_term"];
        for k in o.keys() {
            if !KNOWN.contains(&k.as_str()) {
                let close: Vec<&str> = KNOWN.iter().copied().filter(|c| edit_distance(c, k) <= (k.len().max(c.len()) / 3).max(1)).collect();
                let hint = if close.is_empty() { String::new() } else { format!(" did you mean [{}]?", close.join(", ")) };
                return Err(EsError::x_content_parse(format!("[1:2] [UpdateRequest] unknown field [{k}]{hint}")));
            }
        }
    }
    // Sequence-number preconditions are checked before the update is applied
    // (a no-op update still conflicts).
    let routing = node.index_routing_for(index, req.param("routing"))?;
    if let (Some(sq), Some(pt)) = (cond.if_seq_no, cond.if_primary_term) {
        if let Some(g) = svc.peek(id, routing.as_deref())? {
            if g.seq_no != sq || pt != g.primary_term {
                return Err(EsError::version_conflict(&svc.name, &svc.uuid, id, format!(
                    "version conflict, required seqNo [{sq}], primary term [{pt}]. current document has seqNo [{}] and primary term [{}]", g.seq_no, g.primary_term
                )));
            }
        }
    }
    let retry: usize = req.param("retry_on_conflict").and_then(|s| s.parse().ok()).unwrap_or(0);
    // Only changes whether JSON syntax errors quote the source; ours never do.
    req.bool_param("include_source_on_error")?;
    let (r, src) = svc.update_source(id, &body, raw_source.as_ref(), routing.clone(), retry, cond)?;
    let forced = handle_refresh_doc(req.param("refresh"), &svc, Some((id, routing.as_deref())))?;
    let mut m = write_json(&svc, &r, forced);
    let want_source = match body.get("_source") {
        Some(v) => parse_source_filter(v).ok(),
        None => source_filter_from_params(&req.params),
    };
    let status = if r.result == "created" { 201 } else { 200 };
    if let (Some(f), Some(src)) = (want_source, src) {
        if !matches!(f, SourceFilter::None) {
            let src = match f {
                SourceFilter::Filter { includes, excludes } => elasticsearch_engine::source_json::filter_raw(&src, &includes, &excludes)?,
                _ => src,
            };
            let mut out = serde_json::to_string(&m)?;
            out.pop();
            out.push_str(&format!(",\"get\":{{\"_seq_no\":{},\"_primary_term\":{},\"found\":true,\"_source\":{src}}}}}",r.seq_no,r.primary_term));
            return Ok(Resp::raw(status, out));
        }
    }
    Ok(Resp::json(status, Value::Object(m)))
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + if a[i - 1] == b[j - 1] { 0 } else { 1 });
        }
        prev = cur;
    }
    prev[b.len()]
}

fn error_body(e: &EsError) -> Value {
    let mut v = e.error_object(false);
    if let Some(o) = v.as_object_mut() {
        o.remove("root_cause");
    }
    v
}

fn mget(node: &Node, req: &Req, index: Option<&str>) -> R {
    let body = req.json()?;
    // ES `MultiGetRequest#add`: only `docs` / `ids` at the top level; the
    // defaults come from the URL.
    let specs = crate::mget_body::parse(&body, index, req.param("routing"))?;
    // MultiGetRequest#validate.
    let mut errs: Vec<String> = Vec::new();
    if specs.is_empty() {
        errs.push("no documents to get".into());
    }
    for (i, it) in specs.iter().enumerate() {
        if it.index.is_none() {
            errs.push(format!("index is missing for doc {i}"));
        }
        if it.id.is_none() {
            errs.push(format!("id is missing for doc {i}"));
        }
    }
    if !errs.is_empty() {
        let refs: Vec<&str> = errs.iter().map(String::as_str).collect();
        return Err(EsError::action_validation(&refs));
    }
    let default_source = source_filter_from_params(&req.params);
    let default_stored = req.param("stored_fields").map(|s| docs_stored::list(&json!(s)));
    let mut out = Vec::new();
    for it in specs.into_iter() {
        let (idx, id, routing) = (it.index.unwrap_or_default(), it.id.unwrap_or_default(), it.routing);
        // ES `ShardGetService`: an explicit `_source` (the item's, else the
        // URL's) is fetched alongside `stored_fields`; without one, stored
        // fields alone.
        let explicit = it.source.or_else(|| default_source.clone());
        let filter = explicit.clone().unwrap_or(SourceFilter::All);
        let stored = it.stored.or_else(|| default_stored.clone()).map(|mut sf| {
            if explicit.as_ref().is_some_and(|f| !matches!(f, SourceFilter::None)) && !sf.iter().any(|s| s == "_source") {
                sf.push("_source".to_string());
            }
            sf
        });
        if node.get_index(&idx).is_none() {
            let targets = node.resolve(&idx, elasticsearch_engine::node::ResolveOpts { ignore_unavailable: true, ..Default::default() }).unwrap_or_default();
            if targets.len() > 1 {
                let names: Vec<String> = targets.iter().map(|t| t.svc.name.clone()).collect();
                let e = EsError::illegal_argument(format!("alias [{idx}] has more than one index associated with it [{}], can't execute a single index op", names.join(", ")));
                out.push(json!({"_index": idx, "_id": id, "error": e.error_object(false)}).to_string());
                continue;
            }
        }
        match node.write_index(&idx, false) {
            Err(e) => out.push(json!({"_index": idx, "_id": id, "error": e.error_object(false)}).to_string()),
            Ok(svc) if req.flag("refresh")? && svc.refresh().is_err() => out.push(not_found_json(&svc.name, &id).to_string()),
            Ok(svc) => match access_of(req, &svc).and_then(|acc| secure_get(&svc, &id, &acc, svc.get_routed(&id, routing.as_deref(), req.bool_param("realtime")?.unwrap_or(true))?)).map(|f| exclude_vectors(req, &svc, force_synthetic(req, &svc, f), &filter)) {
                Err(e) => out.push(json!({"_index": svc.name, "_id": id, "error": e.error_object(false)}).to_string()),
                Ok(Some(g)) => match &stored {
                    Some(sf) => out.push(docs_stored::apply(&svc, &get_json(&svc, &id, &g, &filter)?, sf, &g)?),
                    None => out.push(get_json(&svc, &id, &g, &filter)?),
                },
                Ok(None) => out.push(not_found_json(&svc.name, &id).to_string()),
            },
        }
    }
    Ok(Resp::raw(200, format!("{{\"docs\":[{}]}}", out.join(","))))
}

/// A bulk item headed for a data stream: what its failure document needs.
struct FsItem {
    target: elasticsearch_engine::datastream::FailureStoreTarget,
    source: String,
    routing: Option<String>,
    /// The index the item targeted (`document.index`).
    index: String,
    id: Option<String>,
}

/// The bulk item for a failed data stream write: stored in the failure
/// store (`failure_store: used`), or the error with `not_enabled` / `failed`.
fn bulk_failure_item(
    node: &Node,
    req: &Req,
    position: usize,
    f: FsItem,
    shown_index: &str,
    shown_id: Value,
    e: EsError,
    cx: Option<&elasticsearch_engine::datastream::FailureContext>,
    touched: &mut Vec<Arc<IndexService>>,
) -> Map<String, Value> {
    use elasticsearch_engine::datastream::FailureStoreTarget;
    let error_item = |status: &str| {
        let mut m = Map::new();
        m.insert("_index".into(), json!(shown_index));
        m.insert("_id".into(), shown_id.clone());
        m.insert("status".into(), json!(e.status));
        m.insert("failure_store".into(), json!(status));
        m.insert("error".into(), error_body(&e));
        m
    };
    match &f.target {
        FailureStoreTarget::Enabled(ds) => {
            let doc_id = f.id.clone().or_else(|| shown_id.as_str().filter(|s| *s != "auto-generated" && cx.is_none()).map(str::to_string));
            match store_failure(node, req, ds, &f.index, doc_id.as_deref(), f.routing.as_deref(), &f.source, &e, cx, Some(position)) {
                Ok(StoredFailure::Cluster(body)) => {
                    let mut result = body.as_object().cloned().unwrap_or_default();
                    result.insert("failure_store".into(), json!("used"));
                    result.insert("status".into(), json!(201));
                    result
                }
                Ok(StoredFailure::Local(svc, w)) => {
                    let mut m = write_json(&svc, &w, false);
                    m.insert("failure_store".into(), json!("used"));
                    m.insert("status".into(), json!(201));
                    if !touched.iter().any(|s| Arc::ptr_eq(s, &svc)) {
                        touched.push(svc);
                    }
                    m
                }
                Err(_) => error_item("failed"),
            }
        }
        _ => error_item("not_enabled"),
    }
}

/// One bulk action/metadata line (`BulkRequestParser` + its
/// `checkBulkActionIsProperlyClosed`): ES 9 rejects an unclosed object,
/// extra keys after the action and trailing data; `compatible-with=8`
/// requests keep 8.x's lenient parse (first key wins, the rest ignored)
/// with a deprecation warning each.
fn parse_action_line(line: &str, n: usize, compat_v8: bool, warnings: &mut Vec<&'static str>) -> Result<Value, EsError> {
    const UNCLOSED: &str = "A bulk action wasn't closed properly with the closing brace. Malformed objects are currently accepted but will be rejected in a future version.";
    const MULTI: &str = "A bulk action object contained multiple keys. Additional keys are currently ignored but will be rejected in a future version.";
    const TRAILING: &str = "A bulk action contained trailing data after the closing brace. This is currently ignored but will be rejected in a future version.";
    let mut warn = |w: &'static str| {
        if !warnings.contains(&w) {
            warnings.push(w);
        }
    };
    let malformed = || EsError::illegal_argument(format!("Malformed action/metadata line [{n}], expected START_OBJECT or END_OBJECT but found [VALUE_STRING]"));
    let mut stream = serde_json::Deserializer::from_str(line).into_iter::<Value>();
    let v = match stream.next() {
        Some(Ok(v)) => v,
        Some(Err(e)) if e.is_eof() && line.trim_start().starts_with('{') => {
            // The action object (`{"index": {...}`) is missing its closing brace.
            if compat_v8 {
                if let Ok(v) = serde_json::from_str::<Value>(&format!("{line}}}")) {
                    if v.as_object().is_some_and(|o| o.len() == 1) {
                        warn(UNCLOSED);
                        return Ok(v);
                    }
                }
                return Err(malformed());
            }
            let col = line.chars().count() + 1;
            return Err(EsError::new(400, "x_content_e_o_f_exception", format!("[1:{col}] Unexpected end of file")).caused_by(EsError::new(
                400,
                "json_e_o_f_exception",
                format!(
                    "Unexpected end-of-input: expected close marker for Object (start marker at [Source: (byte[])\"{line}\"; line: 1, column: 1])\n at [Source: (byte[])\"{line}\"; line: 1, column: {col}]"
                ),
            )));
        }
        _ => return Err(malformed()),
    };
    let end = stream.byte_offset();
    if let Value::Object(o) = &v {
        if o.len() > 1 {
            if !compat_v8 {
                return Err(EsError::illegal_argument(format!("Malformed action/metadata line [{n}], expected END_OBJECT but found [FIELD_NAME]")));
            }
            warn(MULTI);
            let (k, val) = o.iter().next().map(|(k, v)| (k.clone(), v.clone())).unwrap_or_default();
            return Ok(Value::Object(std::iter::once((k, val)).collect()));
        }
    }
    if !line[end..].trim().is_empty() {
        if !compat_v8 {
            return Err(EsError::illegal_argument(format!("Malformed action/metadata line [{n}], unexpected data after the closing brace")));
        }
        warn(TRAILING);
    }
    Ok(v)
}

/// Indices a bulk wrote with request durability whose translog sync its
/// route runs after the write thread is released ([`sync_deferred`]).
type DeferredSyncs = Arc<std::sync::Mutex<Vec<Arc<IndexService>>>>;

/// The translog syncs (group commits) a bulk deferred, off the write pool:
/// a write thread used to wait out its shards' fsync (one leader syncs,
/// the others park until its sync covers them), holding a write slot
/// without using a core; ES likewise completes a bulk's response when the
/// sync covering its location lands, not on the indexing thread. The
/// response still waits for the sync, and a failed sync fails the request
/// as before.
async fn sync_deferred(services: Vec<Arc<IndexService>>) -> Result<(), EsError> {
    let admission = elasticsearch_engine::rustic_core::admission::Context::capture();
    tokio::task::spawn_blocking(move || {
        let _admission = admission.enter();
        services.iter().try_for_each(|svc| svc.sync_translog())
    }).await.map_err(|e| EsError::internal(format!("translog sync task failed: {e}")))?
}

/// `defer`: collect the indices whose request-durability translog sync the
/// caller runs after this returns (see [`sync_deferred`]); `None` syncs here.
fn bulk(node: &Node, req: &Req, index: Option<&str>, defer: Option<&DeferredSyncs>) -> R {
    let _ingest_user = ingest_user(req);
    let _mapping_authz = mapping_authz(req);
    let start = std::time::Instant::now();
    // Items the cluster coordinator already split (its own shard group):
    // not parsed again. With security on, they were authorized item by
    // item (`security::authz`), the denied ones removed.
    // A coordinator taking typed results (`typed_bulk`): plain writes that
    // succeed are left there instead of rendered here.
    let typed_slot = req.extensions.get::<crate::PreparedBulk>().filter(|p| p.wants_typed()).cloned();
    let prepared = req.extensions.get::<crate::PreparedBulk>().and_then(crate::PreparedBulk::take);
    let body: &str = if prepared.is_some() { "" } else { req.body_str()? };
    let lines: Vec<&str> = body.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l)).collect();
    // Parse action lines first (malformed → whole request fails).
    struct Item {
        action: &'static str,
        meta: Map<String, Value>,
        source: Option<String>,
    }
    let mut items: Vec<Item> = Vec::new();
    let mut lenient_warnings: Vec<&'static str> = Vec::new();
    for (n, p) in prepared.into_iter().flatten().enumerate() {
        let action: &'static str = match p.action.as_str() {
            "index" => "index",
            "create" => "create",
            "update" => "update",
            "delete" => "delete",
            other => return Err(EsError::internal(format!("prepared bulk item [{n}] has the action [{other}]"))),
        };
        // `index` with `op_type: create` is a create (as parsed below).
        let action: &'static str = if action == "index" && p.meta.get("op_type").and_then(Value::as_str) == Some("create") { "create" } else { action };
        items.push(Item { action, meta: p.meta, source: p.source });
    }
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        i += 1;
        if line.is_empty() {
            continue;
        }
        let v = parse_action_line(line, i, req.compat_v8(), &mut lenient_warnings)?;
        // Own the parsed line: its metadata object moves into the item
        // instead of being cloned (one map rebuild per item).
        let o = match v {
            Value::Object(o) => o,
            v => return Err(EsError::illegal_argument(format!("Malformed action/metadata line [{i}], expected START_OBJECT but found [{}]", if v.is_array() { "START_ARRAY" } else { "VALUE_STRING" }))),
        };
        if o.is_empty() {
            return Err(EsError::illegal_argument(format!("Malformed action/metadata line [{i}], expected FIELD_NAME but found [END_OBJECT]")));
        }
        let (a, meta) = o.into_iter().next().unwrap();
        let action: &'static str = match a.as_str() {
            "index" => "index",
            "create" => "create",
            "update" => "update",
            "delete" => "delete",
            other => return Err(EsError::illegal_argument(format!("Malformed action/metadata line [{i}], expected field [create], [delete], [index] or [update] but found [{other}]"))),
        };
        let meta = match meta {
            Value::Object(meta) => meta,
            _ => return Err(EsError::illegal_argument(format!("Malformed action/metadata line [{i}], expected START_OBJECT but found [VALUE_STRING]"))),
        };
        // `index` with `op_type: create` is a create.
        let action: &'static str = if action == "index" && meta.get("op_type").and_then(Value::as_str) == Some("create") { "create" } else { action };
        for k in meta.keys() {
            if !matches!(
                k.as_str(),
                "_index" | "_id" | "routing" | "version" | "version_type" | "if_seq_no" | "if_primary_term" | "pipeline" | "require_alias" | "require_data_stream"
                    | "retry_on_conflict" | "_source" | "dynamic_templates" | "dynamic_template_params" | "list_executed_pipelines" | "op_type"
            ) {
                return Err(EsError::illegal_argument(format!("Action/metadata line [{i}] contains an unknown parameter [{k}]")));
            }
        }
        if let Some(p) = meta.get("dynamic_template_params") {
            if action == "update" {
                return Err(EsError::illegal_argument(format!("Update request in line [{i}] does not accept dynamic_template_params")));
            }
            if !p.is_object() && !p.is_null() {
                return Err(EsError::illegal_argument(format!("Malformed action/metadata line [{i}], expected a simple value for field [dynamic_template_params] but found [{}]", if p.is_array() { "START_ARRAY" } else { "VALUE_STRING" })));
            }
        }
        let source = if action == "delete" {
            None
        } else {
            let s = lines.get(i).copied().unwrap_or("");
            i += 1;
            if s.trim().is_empty() && i > lines.len() {
                return Err(EsError::illegal_argument("The bulk request must be terminated by a newline [\\n]"));
            }
            Some(s.to_string())
        };
        items.push(Item { action, meta, source });
    }
    if items.is_empty() {
        return Err(EsError::action_validation(&["no requests added"]));
    }
    let refresh = req.param("refresh").map(str::to_string);
    let append_only = req.extensions.get::<crate::AppendOnlyItems>();
    // No pipeline exists and the request names none: ingest is skipped.
    let ingest_idle = req.param("pipeline").is_none() && node.ingest_is_idle();
    // Targets without a default / final pipeline, resolved once per request
    // (as ES's bulk action resolves pipelines per request): their items skip
    // the per-document ingest path when neither the request nor the item
    // names a pipeline. Built-in pipelines keep the registry non-empty, so
    // every item of a plain index paid for a pipeline lookup under the
    // ingest registry's lock (the esql track's 10,000-doc bulks contended).
    let mut pipeline_free: HashMap<String, bool> = HashMap::new();
    let mut results: Vec<Option<Value>> = vec![None; items.len()];
    let typed: std::cell::RefCell<Vec<Option<crate::typed_bulk::TypedWrite>>> = std::cell::RefCell::new(if typed_slot.is_some() { vec![None; items.len()] } else { Vec::new() });
    let mut raw_update_sources: HashMap<usize, String> = HashMap::new();
    let mut batch: Vec<(usize, &'static str, WriteReq)> = Vec::new();
    let mut batch_svc: Option<Arc<IndexService>> = None;
    let mut touched: Vec<Arc<IndexService>> = Vec::new();
    let item_status = |r: &str| if r == "created" { 201 } else if r == "not_found" { 404 } else { 200 };
    // Failure store routing of data stream writes, by item position.
    let mut fs_items: HashMap<usize, FsItem> = HashMap::new();
    let mut fs_touched: Vec<Arc<IndexService>> = Vec::new();
    let flush = |batch: &mut Vec<(usize, &'static str, WriteReq)>, svc: &Option<Arc<IndexService>>, results: &mut Vec<Option<Value>>, fs_items: &mut HashMap<usize, FsItem>, fs_touched: &mut Vec<Arc<IndexService>>| {
        let Some(svc) = svc else { return };
        if batch.is_empty() {
            return;
        }
        let (meta, reqs): (Vec<(usize, &'static str)>, Vec<WriteReq>) = batch.drain(..).map(|(p, a, r)| ((p, a), r)).unzip();
        let rs = svc.write(reqs, false);
        let mut typed = typed.borrow_mut();
        // Per flush: the index's identity and `_shards` of `write_json`.
        let typed_index = (!typed.is_empty()).then(|| (Arc::<str>::from(svc.name.as_str()), 1 + svc.setting_usize("index.number_of_replicas", 1), svc.seq_no_disabled()));
        for ((pos, action), r) in meta.into_iter().zip(rs) {
            let fs = fs_items.remove(&pos);
            let r = match (r, &typed_index) {
                (Ok(w), Some((index, total_shards, seq_disabled))) => {
                    let status = item_status(w.result);
                    typed[pos] = Some(crate::typed_bulk::TypedWrite { action, index: index.clone(), id: w.id, version: w.version, result: w.result,
                        seq_no: w.seq_no, primary_term: w.primary_term, total_shards: *total_shards, seq_disabled: *seq_disabled, status, forced_refresh: false });
                    continue;
                }
                (r, _) => r,
            };
            let v = match r {
                Ok(w) => {
                    let mut m = write_json(svc, &w, false);
                    m.insert("status".into(), json!(item_status(w.result)));
                    m
                }
                Err(e) => {
                    let id = e.extra.iter().find(|(k, _)| k == "_id").map(|x| x.1.clone()).or_else(|| fs.as_ref().and(failed_doc_id(&e)).map(Value::String)).unwrap_or(Value::Null);
                    match fs {
                        Some(f) if failure_store_eligible(&e) => bulk_failure_item(node, req, pos, f, &svc.name, id, e, None, fs_touched),
                        _ => {
                            let mut m = Map::new();
                            m.insert("_index".into(), json!(svc.name));
                            m.insert("_id".into(), id);
                            m.insert("status".into(), json!(e.status));
                            m.insert("error".into(), error_body(&e));
                            m
                        }
                    }
                }
            };
            results[pos] = Some(json!({ action: v }));
        }
    };
    let mut ids_for_errors: HashMap<usize, String> = HashMap::new();
    // The last plain concrete index this bulk wrote to
    // (`Node::plain_write_index`): its items skip the alias, data stream
    // and failure store resolutions, which are identities for it, after
    // checking it is still the registered index of that name. A name that
    // is not plain is remembered so it is not probed again; it always
    // takes the full resolution.
    let mut plain: Option<(String, Arc<IndexService>)> = None;
    let mut not_plain: Option<String> = None;
    for (pos, it) in items.into_iter().enumerate() {
        let idx_name = it.meta.get("_index").and_then(Value::as_str).map(str::to_string).or(index.map(str::to_string));
        let id = it.meta.get("_id").map(elasticsearch_engine::query::dsl::scalar_string);
        if let Some(i) = &id {
            ids_for_errors.insert(pos, i.clone());
        }
        let client_id = id.clone();
        let fail = |results: &mut Vec<Option<Value>>, idx: &str, e: EsError| {
            results[pos] = Some(json!({ it.action: {"_index": idx, "_id": id.clone(), "status": e.status, "error": error_body(&e)} }));
        };
        let Some(idx_name) = idx_name else {
            fail(&mut results, "", EsError::action_validation(&["index is missing"]));
            continue;
        };
        if matches!(it.action, "index" | "create") && id.as_deref() == Some("") {
            fail(&mut results, &idx_name, EsError::illegal_argument("if _id is specified it must not be empty"));
            continue;
        }
        let ra = it.meta.get("require_alias").and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s == "true"))).unwrap_or(req.param("require_alias") == Some("true"));
        if it.action != "delete" {
            if let Err(e) = require_alias(node, &idx_name, ra) {
                fail(&mut results, &idx_name, e);
                continue;
            }
            let rds = it.meta.get("require_data_stream").and_then(|v| v.as_bool().or_else(|| v.as_str().map(|s| s == "true"))).unwrap_or(req.param("require_data_stream") == Some("true"));
            if let Err(e) = node.require_data_stream(&idx_name, rds) {
                fail(&mut results, &idx_name, e);
                continue;
            }
        }
        // Ingest pipelines (request / item `pipeline`, index default/final).
        let mut item_source = it.source;
        let routed_plain = plain.as_ref().filter(|(n, svc)| *n == idx_name && node.is_registered_index(n, svc)).map(|(_, svc)| svc.clone());
        let mut item_routing = if routed_plain.is_some() {
            it.meta.get("routing").and_then(Value::as_str).map(str::to_string)
        } else {
            match node.index_routing_for(&idx_name, it.meta.get("routing").and_then(Value::as_str)) {
                Ok(r) => r,
                Err(e) => {
                    fail(&mut results, &idx_name, e);
                    continue;
                }
            }
        };
        let mut item_ingest_dt = None;
        let mut ingested = false;
        let skip_ingest = !it.meta.contains_key("pipeline")
            && (ingest_idle
                || (req.param("pipeline").is_none()
                    && *pipeline_free.entry(idx_name.clone()).or_insert_with(|| matches!(node.index_pipelines(&idx_name), (None, None)))));
        let (idx_name, id) = if matches!(it.action, "index" | "create") && !skip_ingest {
            let pipeline = it.meta.get("pipeline").and_then(Value::as_str).or(req.param("pipeline"));
            ingested = true;
            let source = item_source.take().unwrap_or_default();
            let prepared = req.extensions.get::<crate::PreparedBulkIngestItems>().and_then(|items| items.0.get(&pos));
            let ingest = match prepared {
                Some(crate::PreparedBulkIngest::Write { routing, dynamic_templates }) => Ok(crate::ingest::Prepared::Write {
                    index: idx_name.clone(), id: id.clone(), routing: routing.clone(), source,
                    dynamic_templates: dynamic_templates.clone(),
                }),
                Some(crate::PreparedBulkIngest::Dropped) => Ok(crate::ingest::Prepared::Dropped { index: idx_name.clone(), id: id.clone() }),
                Some(crate::PreparedBulkIngest::Failed(failure)) => Err(Box::new((failure.clone(), source))),
                None => crate::ingest::prepare_ext(node, &idx_name, id.clone(), item_routing.clone(), source, pipeline),
            };
            match ingest {
                Ok(crate::ingest::Prepared::Write { index, id, routing, source, dynamic_templates }) => {
                    item_ingest_dt = dynamic_templates;
                    item_source = Some(source);
                    item_routing = routing;
                    (index, id)
                }
                Ok(crate::ingest::Prepared::Dropped { index, id }) => {
                    let mut m = crate::ingest::dropped_json(&index, id.as_deref());
                    m.insert("status".into(), json!(200));
                    results[pos] = Some(json!({ it.action: m }));
                    continue;
                }
                Err(b) => {
                    let (f, source) = *b;
                    let target = node.failure_store_target(&f.index);
                    if matches!(target, elasticsearch_engine::datastream::FailureStoreTarget::None) {
                        fail(&mut results, &idx_name, f.error);
                    } else {
                        let item = FsItem { target, source, routing: item_routing.clone(), index: idx_name.clone(), id: id.clone() };
                        let shown = json!(id.clone().unwrap_or_else(|| "auto-generated".into()));
                        let m = bulk_failure_item(node, req, pos, item, &idx_name, shown, f.error, Some(&f.context), &mut fs_touched);
                        results[pos] = Some(json!({ it.action: m }));
                    }
                    continue;
                }
            }
        } else {
            (idx_name, id.clone())
        };
        // Without a pipeline the name is the one checked above; after one,
        // check the (possibly new) name again.
        let plain_svc = match routed_plain.filter(|_| !ingested) {
            Some(svc) => Some(svc),
            None => plain.as_ref().filter(|(n, svc)| *n == idx_name && node.is_registered_index(n, svc)).map(|(_, svc)| svc.clone()),
        };
        let is_plain = plain_svc.is_some();
        let svc = match plain_svc {
            Some(svc) => svc,
            None => match node.write_index(&idx_name, it.action != "delete") {
                Ok(s) => {
                    if not_plain.as_deref() != Some(idx_name.as_str()) {
                        match node.plain_write_index(&idx_name) {
                            Some(p) if Arc::ptr_eq(&p, &s) => plain = Some((idx_name.clone(), p)),
                            _ => not_plain = Some(idx_name.clone()),
                        }
                    }
                    s
                }
                Err(e) => {
                    fail(&mut results, &idx_name, e);
                    continue;
                }
            },
        };
        let ds_create = it.action == "create" || (it.action == "index" && id.is_none());
        let ds_src = if matches!(it.action, "index" | "create") { item_source.as_deref() } else { None };
        // A plain index has no failure store (`fs_index` is only read when
        // there is one).
        let fs_index = if is_plain { String::new() } else { failure_store_index(node, req, &idx_name) };
        let fs_target = if matches!(it.action, "index" | "create") && !is_plain { node.failure_store_target(&fs_index) } else { elasticsearch_engine::datastream::FailureStoreTarget::None };
        let internal_backing_write = req.extensions.get::<crate::InternalDataStreamBackingWrite>().is_some();
        // The document's `@timestamp`, read by the data stream check (one
        // scan of the source serves the checks and the routing below).
        let mut probed_ts: Option<serde_json::Value> = None;
        if let Err(e) = (if is_plain { Ok(()) } else if matches!(it.action, "index" | "create") && !internal_backing_write { node.check_backing_write(&idx_name, it.action == "create", it.meta.contains_key("if_seq_no") || it.meta.contains_key("if_primary_term")) } else { Ok(()) }).and_then(|_| node.check_data_stream_write_ts(&idx_name, ds_create, ds_src).map(|ts| { probed_ts = ts; })) {
            if !matches!(fs_target, elasticsearch_engine::datastream::FailureStoreTarget::None) && failure_store_eligible(&e) {
                // Routing (`timestamp_error`) fails on the data stream, before a backing index is chosen.
                let shown = if e.kind == "timestamp_error" { idx_name.clone() } else { svc.name.clone() };
                let item = FsItem { target: fs_target, source: item_source.unwrap_or_default(), routing: item_routing.clone(), index: fs_index.clone(), id: id.clone() };
                let m = bulk_failure_item(node, req, pos, item, &shown, json!(id), e, None, &mut fs_touched);
                results[pos] = Some(json!({ it.action: m }));
            } else {
                fail(&mut results, &svc.name, e);
            }
            continue;
        }
        if !matches!(fs_target, elasticsearch_engine::datastream::FailureStoreTarget::None) {
            // Keep what a failure document needs (the source only when it can be stored).
            let source = if matches!(fs_target, elasticsearch_engine::datastream::FailureStoreTarget::Enabled(_)) { item_source.clone().unwrap_or_default() } else { String::new() };
            fs_items.insert(pos, FsItem { target: fs_target, source, routing: item_routing.clone(), index: fs_index.clone(), id: id.clone() });
        }
        // A time series data stream writes into the backing index covering
        // the document's `@timestamp`.
        let svc = match (matches!(it.action, "index" | "create"), ds_src) {
            (true, Some(_)) if !is_plain => node.time_series_write_index_at(&idx_name, probed_ts.as_ref()).unwrap_or(svc),
            _ => svc,
        };
        if batch_svc.as_ref().map(|s| !Arc::ptr_eq(s, &svc)).unwrap_or(false) || it.action == "update" {
            flush(&mut batch, &batch_svc, &mut results, &mut fs_items, &mut fs_touched);
        }
        if !touched.iter().any(|s| Arc::ptr_eq(s, &svc)) {
            touched.push(svc.clone());
        }
        let cond = match cond_from_meta(&it.meta) {
            Ok(mut c) => {
                if it.action == "create" {
                    c.create = true;
                    // The coordinator generated this id for this very
                    // request (first attempt, in process): append-only.
                    // (An ingest pipeline that set another `_id` voids it.)
                    c.append_only = append_only.is_some_and(|a| a.0.get(pos).copied().unwrap_or(false)) && id.is_some() && id == client_id;
                }
                c
            }
            Err(e) => {
                fail(&mut results, &svc.name, e);
                continue;
            }
        };
        let routing = item_routing;
        match it.action {
            "index" | "create" => {
                // Bulk `dynamic_templates` (path -> template name): written on
                // its own so the request's templates apply to this document only.
                // Templates set by an ingest pipeline replace the request's.
                let dt: Option<std::collections::HashMap<String, String>> = item_ingest_dt.take().filter(|m| !m.is_empty()).or_else(|| {
                    it.meta.get("dynamic_templates").and_then(Value::as_object).filter(|m| !m.is_empty()).map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect())
                });
                let dtp: Option<elasticsearch_engine::document::TemplateParams> = it.meta.get("dynamic_template_params").and_then(Value::as_object).filter(|m| !m.is_empty()).map(|m| {
                    m.iter()
                        .map(|(k, v)| (k.clone(), v.as_object().map(|o| o.iter().map(|(pk, pv)| (pk.clone(), pv.as_str().map(str::to_string).unwrap_or_else(|| pv.to_string()))).collect()).unwrap_or_default()))
                        .collect()
                });
                batch_svc = Some(svc.clone());
                batch.push((pos, it.action, WriteReq::Index { id: id.clone(), source: item_source.unwrap_or_default(), routing, cond }));
                if dt.is_some() || dtp.is_some() {
                    let last = batch.pop().expect("just pushed");
                    flush(&mut batch, &batch_svc, &mut results, &mut fs_items, &mut fs_touched);
                    let mut one = vec![last];
                    elasticsearch_engine::document::with_dynamic_templates(dt.unwrap_or_default(), || {
                        elasticsearch_engine::document::with_dynamic_template_params(dtp.unwrap_or_default(), || flush(&mut one, &batch_svc, &mut results, &mut fs_items, &mut fs_touched))
                    });
                }
            }
            "delete" => {
                let Some(id) = id.clone() else {
                    fail(&mut results, &svc.name, EsError::action_validation(&["id is missing"]));
                    continue;
                };
                batch_svc = Some(svc.clone());
                batch.push((pos, "delete", WriteReq::Delete { id, routing, cond }));
            }
            _ => {
                let Some(id) = id.clone() else {
                    fail(&mut results, &svc.name, EsError::action_validation(&["id is missing"]));
                    continue;
                };
                let (body,raw_source) = match elasticsearch_engine::source_json::update_request(item_source.as_deref().unwrap_or("")) {
                    Ok(v) => v,
                    Err(e) => {
                        fail(&mut results, &svc.name, EsError::json_parse(&e));
                        continue;
                    }
                };
                // `ShardBulkInferenceActionFilter` checks on update items.
                if svc.lookup().fields.values().any(|f| f.declared.is_semantic()) {
                    if body.get("script").is_some() {
                        fail(&mut results, &svc.name, EsError::new(400, "status_exception", "Cannot apply update with a script on indices that contain [semantic_text] field(s)"));
                        continue;
                    }
                    if svc.setting("index.mapping.semantic_text.use_legacy_format").as_deref() == Some("true") {
                        if let Some(d) = body.get("doc") {
                            if let Err(e) = elasticsearch_engine::inference::semantic::check_legacy_bulk_update(&svc.lookup(), d) {
                                fail(&mut results, &svc.name, e);
                                continue;
                            }
                        }
                    }
                }
                let retry = it.meta.get("retry_on_conflict").and_then(Value::as_u64).unwrap_or(0) as usize;
                let mut c = WriteCond::default();
                c.if_seq_no = cond.if_seq_no;
                c.if_primary_term = cond.if_primary_term;
                match svc.update_source(&id, &body, raw_source.as_ref(), routing, retry, c) {
                    Ok((w, src)) => {
                        let mut m = write_json(&svc, &w, false);
                        m.insert("status".into(), json!(item_status(w.result)));
                        let want = match body.get("_source").or_else(|| it.meta.get("_source")) {
                            Some(v) => parse_source_filter(v).ok(),
                            None => source_filter_from_params(&req.params),
                        };
                        if let (Some(f), Some(src)) = (want, src) {
                            if !matches!(f, SourceFilter::None) {
                                let src = match f {
                                    SourceFilter::Filter { includes, excludes } => elasticsearch_engine::source_json::filter_raw(&src, &includes, &excludes)?,
                                    _ => src,
                                };
                                m.insert("get".into(), json!({"_seq_no":w.seq_no,"_primary_term":w.primary_term,"found":true}));
                                if let Ok(v)=serde_json::from_str::<Value>(&src) { m.get_mut("get").unwrap().as_object_mut().unwrap().insert("_source".into(),v); }
                                else { raw_update_sources.insert(pos, src); }
                            }
                        }
                        results[pos] = Some(json!({"update": m}));
                    }
                    Err(e) => fail(&mut results, &svc.name, e),
                }
            }
        }
    }
    flush(&mut batch, &batch_svc, &mut results, &mut fs_items, &mut fs_touched);
    for svc in fs_touched {
        if !touched.iter().any(|s| Arc::ptr_eq(s, &svc)) {
            touched.push(svc);
        }
    }
    for svc in &touched {
        if sync_mode(svc) {
            match defer {
                Some(defer) => defer.lock().unwrap_or_else(|e| e.into_inner()).push(svc.clone()),
                None => svc.sync_translog()?,
            }
        }
        if refresh.is_some() {
            let forced = handle_refresh(refresh.as_deref(), svc)?;
            if forced {
                for w in typed.borrow_mut().iter_mut().flatten() {
                    if *w.index == *svc.name { w.forced_refresh = true; }
                }
                for r in results.iter_mut().flatten() {
                    if let Some(o) = r.as_object_mut().and_then(|o| o.values_mut().next()).and_then(Value::as_object_mut) {
                        if o.get("_index").and_then(Value::as_str) == Some(svc.name.as_str()) && o.get("error").is_none() {
                            o.insert("forced_refresh".into(), json!(true));
                        }
                    }
                }
            }
        }
    }
    // Errors from the shard lack the id; fill it from the action line.
    for (pos, r) in results.iter_mut().enumerate() {
        if let Some(o) = r.as_mut().and_then(|v| v.as_object_mut()).and_then(|o| o.values_mut().next()).and_then(Value::as_object_mut) {
            if o.get("_id").map(Value::is_null).unwrap_or(false) {
                if let Some(id) = ids_for_errors.get(&pos) {
                    o.insert("_id".into(), json!(id));
                }
            }
        }
    }
    let headers: Vec<(&'static str, String)> = lenient_warnings
        .iter()
        .map(|w| ("Warning", format!("299 Elasticsearch-{}-{} \"{w}\"", elasticsearch_engine::node::ES_VERSION, crate::cluster::BUILD_HASH)))
        .collect();
    let mut typed = typed.into_inner();
    if let (Some(slot), true) = (&typed_slot, raw_update_sources.is_empty()) {
        let items = results.into_iter().zip(typed.into_iter().chain(std::iter::repeat(None))).map(|(json, write)| match write {
            Some(w) => crate::typed_bulk::TypedItem::Write(w),
            None => crate::typed_bulk::TypedItem::Json(json.unwrap_or(Value::Null)),
        }).collect();
        let answer = crate::typed_bulk::TypedBulk { took: start.elapsed().as_millis() as u64, items };
        return match slot.put_typed(answer) {
            // The coordinator renders the items; this body is not read.
            Ok(()) => Ok(Resp { status: 200, body: RespBody::Json(json!({"took": 0, "errors": false, "items": []})), headers }),
            Err(answer) => Ok(Resp { status: 200, body: RespBody::Json(answer.to_value()), headers }),
        };
    }
    // Not taken typed (raw update sources): every write as the JSON path's.
    for (r, w) in results.iter_mut().zip(typed.iter_mut()) {
        if let Some(w) = w.take() { *r = Some(w.to_value(w.action)); }
    }
    let errors = results.iter().flatten().any(|r| r.as_object().and_then(|o| o.values().next()).map(|v| v.get("error").is_some()).unwrap_or(false));
    let items: Vec<Value> = results.into_iter().map(|r| r.unwrap_or(Value::Null)).collect();
    if raw_update_sources.is_empty() {
        return Ok(Resp {status:200,body:RespBody::Json(json!({"took":start.elapsed().as_millis() as u64,"errors":errors,"items":items})),headers});
    }
    let mut response = elasticsearch_engine::source_json::Source::from_value(&json!({"took":start.elapsed().as_millis() as u64,"errors":errors,"items":items}));
    for (pos, raw) in raw_update_sources {
        response.insert_path(&["items", &pos.to_string(), "update", "get", "_source"], elasticsearch_engine::source_json::Source::parse(&raw)?);
    }
    Ok(Resp { status:200, body:RespBody::Raw(response.text()), headers })
}
