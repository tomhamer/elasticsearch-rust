//! `_reindex` over multi-shard indices.
//!
//! The engine executes a reindex inside one engine: it searches the source
//! indices it holds and writes into its own copy of the destination. Under
//! the cluster that engine is a node's shard-0 engine, so a multi-shard
//! source was read only from its shard 0 and a multi-shard destination got
//! every document in shard 0 regardless of routing.
//!
//! When the source or the destination has more than one shard the reindex
//! runs here instead, through the ordinary cluster paths: a scroll over the
//! source (fanned out to every shard) and `_bulk` writes into the
//! destination (routed per document to its shard's primary, replicated like
//! any other write). Script, routing, versioning, `max_docs`, `conflicts`
//! and failure accounting mirror the engine's `reindex::execute`.
//!
//! `wait_for_completion=false` runs the same reindex as a cluster task
//! (`front::bg_task`): `{"task": "node:n"}` now, the BulkByScroll response
//! under `_tasks/node:n` when done. A destination that does not exist yet
//! counts as multi-shard when the index template that would create it sets
//! `number_of_shards > 1`.

use crate::front::{await_mirror, dispatch, RestReq, RestResp};
use crate::node::ClusterNode;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::reindex::{parse_reindex, response_status, ByQueryRequest, Kind, PageCheck, Stats};
use elasticsearch_engine::script::update::{run_doc_script_source, DocMeta, DocOp};
use elasticsearch_engine::shard::VersionType;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// A sub-request of the reindex, authorized as its caller (`creds`: the
/// caller's credential headers; ES runs reindex in the caller's context).
fn internal(creds: &[(String, String)], method: &str, uri: std::fmt::Arguments<'_>, body: impl AsRef<[u8]>, ndjson: bool) -> elasticsearch_engine::error::Result<RestReq> {
    let ct = if ndjson { "application/x-ndjson" } else { "application/json" };
    let mut req=RestReq::try_new_uri(method,uri,creds,body.as_ref(),0,Default::default(),None)?;
    req.push_header("content-type",ct)?;
    // The reindex's own bulks may carry its internal action metadata.
    if ndjson { req.push_header(elasticsearch_http::util::REINDEX_BULK_HEADER, "1")?; }
    Ok(req)
}

/// Bulk delete metadata of a reindex's scripted delete: ES's internal
/// version equality (see `elasticsearch_http::util::REINDEX_VERSION_EQ`).
const REINDEX_VERSION_EQ: &str = elasticsearch_http::util::REINDEX_VERSION_EQ;

/// The caller's credential headers (or its `_system` identity).
pub(crate) fn credentials(req: &RestReq) -> Result<crate::front::SharedHeaders,EsError> {
    crate::front::SharedHeaders::try_filtered(&req.headers,|(name,_)|name.eq_ignore_ascii_case("authorization")||name.eq_ignore_ascii_case("es-security-runas-user")||name.eq_ignore_ascii_case(crate::front::SYSTEM_HEADER)||name.eq_ignore_ascii_case(crate::front::IDENTITY_HEADER))
}

/// `dispatch` behind a boxed future (`dispatch` is what calls in here).
fn send(node: &Arc<ClusterNode>, req: elasticsearch_engine::error::Result<RestReq>) -> std::pin::Pin<Box<dyn std::future::Future<Output = RestResp> + Send>> {
    let node = node.clone();
    Box::pin(async move { match req { Ok(req) => dispatch(&node, req).await, Err(e) => RestResp::json(e.status, &e.to_json()) } })
}

fn call(node: &Arc<ClusterNode>, creds: &[(String, String)], method: &str, uri: std::fmt::Arguments<'_>, body: Option<Value>) -> std::pin::Pin<Box<dyn std::future::Future<Output=RestResp>+Send>> {
    let request=internal(creds,method,uri,Vec::new(),false).and_then(|mut req| {if let Some(body)=body {req.set_json(&body)?;} Ok(req)});
    send(node,request)
}

fn err_resp(e: &EsError) -> RestResp {
    RestResp::json(e.status, &e.to_json_with_trace(false))
}

/// The reindex when it involves a multi-shard index, `None` to leave it to
/// the engine path (single-shard indices, requests the engine rejects or
/// renders an error for).
pub async fn maybe(node: &Arc<ClusterNode>, req: &RestReq) -> Option<RestResp> {
    let params = req.params();
    let wait = params.get("wait_for_completion").map(|v| v != "false").unwrap_or(true);
    let text = std::str::from_utf8(&req.body).ok()?.to_string();
    let (body,raw)=match req.json() {Some(body)=>(body,None),None=>elasticsearch_engine::source_json::update_request(&text).ok()?};
    let mut r = parse_reindex(&body, &text, &params).ok()?;
    r.raw_script=raw.as_ref().and_then(|r|r.get("script")).cloned();
    let dest = r.dest_index.clone()?;
    await_mirror(node).await;
    // A remote source (ES `RemoteScrollableHitSource`) is read from the
    // remote cluster, whatever indices of the same name exist here. With
    // one destination shard (and no possible redirection) the engine path
    // runs it on the destination primary; otherwise the reindex runs here,
    // reading the remote source and writing through the cluster.
    if r.remote.is_some() {
        let distributed = { let st = node.state(); dest_shards(node, &dest) > 1 || may_redirect(&st, node, &r, &dest) };
        if !distributed {
            return None;
        }
        if let Some(spec) = &r.remote {
            if let Err(e) = spec.check_whitelist(&elasticsearch_http::remote_whitelist(&node.engine)) {
                return Some(err_resp(&e));
            }
        }
        let creds=match credentials(req) {Ok(v)=>v,Err(e)=>return Some(err_resp(&e))};
        if !wait {
            let (n, desc, rps) = (node.clone(), r.description(), r.rps);
            let work = async move { run(&n, &creds, &r, &body, &dest).await };
            return Some(crate::front::bg_task::spawn(node, req, Kind::Reindex.action(), desc, rps, work));
        }
        return Some(run(node, &creds, &r, &body, &dest).await);
    }
    let engine = node.engine.clone();
    let (expr, opts) = (r.source.clone(), r.resolve);
    let sources = tokio::task::spawn_blocking(move || engine.resolve(&expr, opts).map(|ts| ts.iter().map(|t| t.svc.name.clone()).collect::<BTreeSet<_>>())).await.ok()?.ok()?;
    if sources.is_empty() {
        return None;
    }
    let multi = {
        let st = node.state();
        let md = &st.metadata;
        sources.iter().any(|s| crate::shard_engines::shards_of(md, s) > 1) || dest_shards(node, &dest) > 1
            // The engine path runs on the destination primary's node and reads
            // the source from that node only.
            || !colocated(&st, &sources, &dest)
            // A script or ingest pipeline can send documents to another
            // index than `dest`: every write must go through the primary of
            // the index it lands in (and wait for its replicas), as ES's
            // bulk action routes each item.
            || may_redirect(&st, node, &r, &dest)
    };
    if !multi {
        return None;
    }
    let creds=match credentials(req) {Ok(v)=>v,Err(e)=>return Some(err_resp(&e))};
    if !wait {
        let (n, desc, rps) = (node.clone(), r.description(), r.rps);
        let work = async move { run(&n, &creds, &r, &body, &dest).await };
        return Some(crate::front::bg_task::spawn(node, req, Kind::Reindex.action(), desc, rps, work));
    }
    Some(run(node, &creds, &r, &body, &dest).await)
}

/// Whether a document of the reindex can land in another index than
/// `dest`: a script (`ctx._index`), a request pipeline, or a default /
/// final pipeline of the destination.
fn may_redirect(st: &elasticsearch_cluster::state::ClusterState, node: &ClusterNode, r: &ByQueryRequest, dest: &str) -> bool {
    if r.script.is_some() || r.pipeline.as_deref().is_some_and(|p| p != "_none") {
        return true;
    }
    let name = if st.metadata.indices.contains_key(dest) { Some(dest.to_string()) } else { node.engine.write_index(dest, false).ok().map(|svc| svc.name.clone()) };
    let Some(im) = name.as_deref().and_then(|n| st.metadata.indices.get(n)) else { return false };
    ["index.default_pipeline", "index.final_pipeline"].iter().any(|k| im.settings.get(*k).and_then(Value::as_str).is_some_and(|p| !p.is_empty() && p != "_none"))
}

/// Whether every source shard has a started copy on the node holding the
/// destination's primary (an index that does not exist yet has none).
fn colocated(st: &elasticsearch_cluster::state::ClusterState, sources: &BTreeSet<String>, dest: &str) -> bool {
    let started_on = |index: &str, shard: u32, node: Option<&str>| {
        st.routing_table.shard_table(index, shard).and_then(|t| t.copies.iter().find(|c| c.active() && (node.is_none() && c.primary || node.is_some() && c.node_id.as_deref() == node)))
            .and_then(|c| c.node_id.clone())
    };
    let Some(home) = started_on(dest, 0, None) else { return false };
    sources.iter().all(|s| (0..crate::shard_engines::shards_of(&st.metadata, s)).all(|k| started_on(s, k, Some(&home)).is_some()))
}

/// Shard count of the destination's write index (1 when it does not exist).
fn dest_shards(node: &ClusterNode, dest: &str) -> u32 {
    let st = node.state();
    let md = &st.metadata;
    if md.indices.contains_key(dest) {
        return crate::shard_engines::shards_of(md, dest);
    }
    // An alias / data stream: the engine knows its write index.
    match node.engine.write_index(dest, false) {
        Ok(svc) => crate::shard_engines::shards_of(md, &svc.name),
        // Created by the reindex itself: the template that would apply.
        Err(_) => {
            let settings = node.engine.template_for(dest).0;
            let n = &settings["index"]["number_of_shards"];
            n.as_u64().or_else(|| n.as_str().and_then(|s| s.parse().ok())).map(|n| n.max(1) as u32).unwrap_or(1)
        }
    }
}

fn source_page(bytes:&[u8])->Result<(Value,Vec<Option<String>>),EsError>{
    use elasticsearch_engine::source_json::Source;
    if let Ok(v)=serde_json::from_slice::<Value>(bytes){return Ok((v,vec![]));}
    let mut page=Source::parse(std::str::from_utf8(bytes).map_err(|e|EsError::internal(e.to_string()))?)?;
    let mut sources=Vec::new();
    if let Source::Object(root)=&mut page {
        if let Some(Source::Object(hits))=root.get_mut("hits") {
            if let Some(Source::Array(hits))=hits.get_mut("hits") {
                for hit in hits {
                    let src=if let Source::Object(h)=hit {h.remove("_source").map(|s|s.text())}else{None};
                    sources.push(src);
                }
            }
        }
    }
    Ok((serde_json::from_str(&page.text())?,sources))
}

async fn run(node: &Arc<ClusterNode>, creds: &[(String, String)], r: &ByQueryRequest, body: &Value, dest: &str) -> RestResp {
    let start = Instant::now();
    let mut st = Stats::default();
    let data = node.engine.data.clone();
    // Fail like the engine (before anything executes) on a bad script.
    if let Some(s) = &r.script {
        if let Err(e) = elasticsearch_engine::script::compile_json_source(s, r.raw_script.as_ref(), elasticsearch_engine::script::ScriptContext::Reindex, Some(&elasticsearch_engine::script::stored::for_data_dir(&data))) {
            return err_resp(&e);
        }
    }
    let page_size = r.max_docs.map(|m| (m as usize).min(r.batch_size)).unwrap_or(r.batch_size).max(1);
    let mut sbody = json!({"version": true, "seq_no_primary_term": true, "sort": ["_doc"], "track_total_hits": true});
    if let Some(q) = &r.query {
        sbody["query"] = q.clone();
    }
    // The complete source, vectors included (ES reindex reads sources with
    // `exclude_vectors: false` unless the request says otherwise).
    sbody["_source"] = reindex_source_filter(body["source"].get("_source"), r.source_exclude_vectors);
    // A manual slice reads only its partition (ES `source.slice`).
    if let Some((id, max)) = r.slice {
        sbody["slice"] = json!({"id": id, "max": max});
    }
    let remote = match &r.remote {
        Some(spec) => match spec.with_node_tls(&node.engine.node_settings.read()) {
            Ok(spec) => Some(Arc::new(spec)),
            Err(e) => return err_resp(&e),
        },
        None => None,
    };
    // The source expression itself, with the request's index options (ES
    // reindex searches `source.index`): aliases keep their filter and
    // `search_routing`. A partial page would silently drop the documents
    // of the shards that failed: the search must be complete (ES
    // BulkByScroll requests never allow partial results), and every page
    // is checked again below.
    let src = source_expression(&r.source);
    let opts = resolve_params(&r.resolve);
    let first = match &remote {
        Some(spec) => {
            // Version probe, as the remote reindex does before searching.
            let probe = remote_call(spec.clone(), "GET", "/".to_string(), None).await;
            if !probe.is_success() {
                return probe;
            }
            let mut rbody = serde_json::Map::new();
            if let Some(q) = &r.query {
                rbody.insert("query".into(), q.clone());
            }
            if let Some(s) = &r.remote_source {
                rbody.insert("_source".into(), s.clone());
            }
            let path = format!("/{src}/_search?scroll={}&sort=_doc%3Aasc&size={page_size}&allow_partial_search_results=false&version=true", r.scroll);
            remote_call(spec.clone(), "POST", path, Some(Value::Object(rbody).to_string())).await
        }
        None => {
            // Finish admitted URI construction before suspension: fmt::Arguments is not Send.
            let prepared_call = call(node, creds, "POST", format_args!("/{src}/_search?scroll=5m&size={page_size}&allow_partial_search_results=false&{opts}"), Some(sbody));
            prepared_call.await
        }
    };
    if !first.is_success() {
        return first;
    }
    let (mut page, mut raw_sources) = match source_page(&first.body) { Ok(p)=>p, Err(e)=>return err_resp(&e) };
    st.total = {
        let t = page["hits"]["total"]["value"].as_u64().unwrap_or(0);
        r.max_docs.map(|m| m.min(t)).unwrap_or(t)
    };
    let mut scroll_id: Option<String> = None;
    let mut touched: BTreeSet<String> = BTreeSet::new();
    let mut error: Option<RestResp> = None;
    loop {
        if let Some(id) = page["_scroll_id"].as_str() {
            scroll_id = Some(id.to_string());
        }
        // ES `AbstractAsyncBulkByScrollAction.onScrollResponse`: a page with
        // shard failures or a timeout ends the run with them; its hits are
        // not processed.
        let check = PageCheck::of(&page);
        if !check.complete() {
            st.timed_out |= check.timed_out;
            st.failures.extend(check.failures);
            break;
        }
        let mut hits = page["hits"]["hits"].as_array().cloned().unwrap_or_default();
        if let Some(m) = r.max_docs {
            // Successfully processed documents count toward `max_docs`
            // (ES): conflicts skipped with `conflicts: proceed` do not.
            hits.truncate(m.saturating_sub(st.successful()) as usize);
        }
        if hits.is_empty() {
            break;
        }
        // `POST _tasks/<id>/_cancel` on the cluster task: stop between batches
        // like the engine's BulkByScroll worker.
        if crate::front::bg_task::current().map(|t| t.is_cancelled()).unwrap_or(false) {
            st.canceled = Some("by user request".into());
            break;
        }
        let batch_start = Instant::now();
        let docs = hits.len();
        st.batches += 1;
        let actions = match build_actions_source(r, dest, &hits, &raw_sources, &data, &mut st) {
            Ok(a) => a,
            Err(e) => {
                error = Some(err_resp(&e));
                break;
            }
        };
        let mut abort = false;
        if !actions.is_empty() {
            for a in &actions {
                touched.insert(a.index.clone());
            }
            match bulk_retrying(&actions, &mut st.retries_bulk, |nd| { let prepared_request = internal(creds, "POST", format_args!("/_bulk"), nd, true); send(node, prepared_request) }).await {
                Ok(items) => abort = tally(r, &actions, &items, &mut st),
                Err(resp) => {
                    error = Some(resp);
                    break;
                }
            }
        }
        if abort || r.max_docs.map(|m| st.successful() >= m).unwrap_or(false) {
            break;
        }
        // `requests_per_second` (the task's current rate: `_rethrottle`
        // applies from the next batch on).
        throttle(r.rps, batch_start, docs, &mut st).await;
        let Some(id) = scroll_id.clone() else { break };
        let next = match &remote {
            Some(spec) => remote_call(spec.clone(), "POST", "/_search/scroll".to_string(), Some(json!({"scroll": r.scroll, "scroll_id": id}).to_string())).await,
            None => {
                let prepared_call = call(node, creds, "POST", format_args!("/_search/scroll"), Some(json!({"scroll": "5m", "scroll_id": id})));
                prepared_call.await
            }
        };
        if !next.is_success() {
            error = Some(next);
            break;
        }
        match source_page(&next.body) { Ok((p,s))=>{page=p;raw_sources=s;}, Err(e)=>{error=Some(err_resp(&e));break;} }
    }
    if let Some(id) = scroll_id {
        match &remote {
            Some(spec) => {
                let _ = remote_call(spec.clone(), "DELETE", "/_search/scroll".to_string(), Some(json!({"scroll_id": [id]}).to_string())).await;
            }
            None => {
                let prepared_call = call(node, creds, "DELETE", format_args!("/_search/scroll"), Some(json!({"scroll_id": id})));
                let _ = prepared_call.await;
            }
        }
    }
    if let Some(e) = error {
        return e;
    }
    if r.refresh && !touched.is_empty() {
        let names = crate::front::UriJoin(touched.iter(),",");
        let prepared_call = call(node, creds, "POST", format_args!("/{names}/_refresh"), None);
        let _ = prepared_call.await;
    }
    let (status, v) = response(&st, r.rps, start.elapsed().as_millis());
    RestResp::json(status, &v)
}

/// NDJSON of the bulk `actions` at `positions`.
fn bulk_body(actions: &[Action], positions: &[usize]) -> Result<axum::body::Bytes,EsError> {
    RestReq::encode_body(|out| {
        for &i in positions {
            let a=&actions[i];serde_json::to_writer(&mut *out,&a.action)?;out.write_all(b"\n")?;
            if let Some(source)=&a.source {out.write_all(source.as_bytes())?;out.write_all(b"\n")?;}
        } Ok(())
    })
}

/// Wait until `docs` at the current rate have had their time since
/// `batch_start` (ES `WorkerBulkByScrollTaskState.delayPrepareBulkRequest`).
/// The rate is re-read while waiting, so a rethrottle takes effect at once;
/// a cancelled task stops waiting.
async fn throttle(rps: f64, batch_start: Instant, docs: usize, st: &mut Stats) {
    loop {
        let task = crate::front::bg_task::current();
        if task.as_ref().is_some_and(|t| t.is_cancelled()) {
            return;
        }
        let rps = task.as_ref().map(|t| t.rps()).unwrap_or(rps);
        if rps.is_nan() || rps <= 0.0 || rps.is_infinite() {
            return;
        }
        let target = batch_start + std::time::Duration::from_secs_f64(docs as f64 / rps);
        let now = Instant::now();
        if now >= target {
            return;
        }
        let nap = (target - now).min(std::time::Duration::from_millis(50));
        tokio::time::sleep(nap).await;
        st.throttled_micros += nap.as_micros() as u64;
    }
}

fn cancelled() -> bool {
    crate::front::bg_task::current().map(|t| t.is_cancelled()).unwrap_or(false)
}

/// ES `BulkByScrollTask` retries: a rejected bulk (429 as a whole) or
/// rejected items (429) are re-sent with ES's exponential backoff, at most
/// `BULK_MAX_RETRIES` times, the wait ending early when the task is
/// cancelled (the last answers then stand). Each retry round counts in
/// `retries`. The items in action order, or the failed response.
async fn bulk_retrying<F, Fut>(actions: &[Action], retries: &mut u64, mut send: F) -> Result<Vec<Value>, RestResp>
where
    F: FnMut(axum::body::Bytes) -> Fut,
    Fut: std::future::Future<Output = RestResp>,
{
    use elasticsearch_engine::reindex::{bulk_retry_delay, BULK_MAX_RETRIES};
    let mut items: Vec<Value> = vec![Value::Null; actions.len()];
    let mut pending: Vec<usize> = (0..actions.len()).collect();
    for attempt in 0..=BULK_MAX_RETRIES {
        let body=bulk_body(actions,&pending).map_err(|e|err_resp(&e))?;
        let resp = send(body).await;
        let last = attempt == BULK_MAX_RETRIES;
        let again: Vec<usize> = if resp.status == 429 && !last {
            pending.clone()
        } else if !resp.is_success() {
            return Err(resp);
        } else {
            let got = resp.body_json().map(|v| v["items"].as_array().cloned().unwrap_or_default()).unwrap_or_default();
            let mut again = Vec::new();
            for (&i, item) in pending.iter().zip(got) {
                let status = item.as_object().and_then(|o| o.values().next()).and_then(|v| v["status"].as_u64()).unwrap_or(0);
                if status == 429 { again.push(i); }
                items[i] = item;
            }
            again
        };
        if again.is_empty() || last {
            break;
        }
        *retries += 1;
        let until = Instant::now() + bulk_retry_delay(attempt);
        let mut waited = true;
        while Instant::now() < until {
            if cancelled() { waited = false; break; }
            tokio::time::sleep((until - Instant::now()).min(std::time::Duration::from_millis(50))).await;
        }
        if !waited {
            break;
        }
        pending = again;
    }
    Ok(items)
}

/// The engine's synchronous reindex response (`took`, `timed_out`, the
/// BulkByScroll status, `failures`); status = the worst failure's, else 200.
pub(crate) fn response(st: &Stats, rps: f64, took: u128) -> (u16, Value) {
    let mut m = serde_json::Map::new();
    m.insert("took".into(), json!(took as u64));
    m.insert("timed_out".into(), json!(st.timed_out));
    m.extend(st.status_json(Kind::Reindex, rps, 0, None));
    m.insert("failures".into(), Value::Array(st.failures.iter().map(|f| f.1.clone()).collect()));
    (response_status(&st.failures, st.timed_out), Value::Object(m))
}

/// A call to the remote source cluster (blocking client, off the runtime
/// workers), as a response.
async fn remote_call(spec: Arc<elasticsearch_engine::remote::RemoteSpec>, method: &'static str, path: String, body: Option<String>) -> RestResp {
    let out = tokio::task::spawn_blocking(move || spec.call(method, &path, body.as_deref())).await;
    match out {
        Ok(Ok(text)) => RestResp::new(200, vec![], text.as_bytes()),
        Ok(Err(e)) => err_resp(&e),
        Err(e) => err_resp(&EsError::internal(format!("remote source call failed: {e}"))),
    }
}

/// The `_source` of the reindex source search: the request's filter with
/// vectors included unless `exclude_vectors` asks otherwise.
fn reindex_source_filter(user: Option<&Value>, exclude_vectors: Option<bool>) -> Value {
    let ex = json!(exclude_vectors.unwrap_or(false));
    match user {
        Some(Value::Bool(false)) => json!(false),
        None | Some(Value::Bool(true)) | Some(Value::Null) => json!({"exclude_vectors": ex}),
        Some(Value::Object(o)) => {
            let mut o = o.clone();
            o.entry("exclude_vectors").or_insert(ex);
            Value::Object(o)
        }
        Some(other) => json!({"includes": other, "exclude_vectors": ex}),
    }
}

/// `source.index` as a URI path component (each comma-separated part
/// encoded).
fn source_expression(source: &str) -> String {
    let source = if source.is_empty() { "_all" } else { source };
    source.split(',').map(crate::front::encode_component).collect::<Vec<_>>().join(",")
}

/// The index options of a resolve as search URI parameters.
fn resolve_params(o: &elasticsearch_engine::node::ResolveOpts) -> String {
    let mut wildcards: Vec<&str> = Vec::new();
    if o.expand_open { wildcards.push("open"); }
    if o.expand_closed { wildcards.push("closed"); }
    if o.expand_hidden { wildcards.push("hidden"); }
    if wildcards.is_empty() { wildcards.push("none"); }
    format!("ignore_unavailable={}&allow_no_indices={}&expand_wildcards={}", o.ignore_unavailable, o.allow_no_indices, wildcards.join(","))
}

/// One bulk action built from a source hit.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Action {
    pub index: String,
    pub id: String,
    pub action: Value,
    pub source: Option<String>,
}

/// The bulk actions for one page of source hits (script noops are counted
/// in `st.noops` and produce no action). Synchronous: a compiled script
/// never lives across an await.
pub(crate) fn build_actions(r: &ByQueryRequest, dest: &str, hits: &[Value], data: &Path, st: &mut Stats) -> Result<Vec<Action>, EsError> {
    build_actions_source(r,dest,hits,&[],data,st)
}
fn build_actions_source(r:&ByQueryRequest,dest:&str,hits:&[Value],raw_sources:&[Option<String>],data:&Path,st:&mut Stats)->Result<Vec<Action>,EsError>{
    let script = match &r.script {
        Some(s) => Some(elasticsearch_engine::script::compile_json_source(s, r.raw_script.as_ref(), elasticsearch_engine::script::ScriptContext::Reindex, Some(&elasticsearch_engine::script::stored::for_data_dir(data)))?),
        None => None,
    };
    let mut out = Vec::with_capacity(hits.len());
    for (pos,h) in hits.iter().enumerate() {
        let Some(hid) = h["_id"].as_str() else { continue };
        let mut index = dest.to_string();
        // ES `Reindexer`: the destination request first (id, `dest.routing`
        // policy, external version), then the script's metadata changes over
        // it: a cleared `_id` generates one (production review #191), a
        // changed `_routing` wins over the policy (#199), and a scripted
        // delete keeps the version condition (#241).
        let source_routing = h["_routing"].as_str().map(str::to_string);
        let source_version = h["_version"].as_u64().unwrap_or(1);
        let mut id: Option<String> = Some(hid.to_string());
        let mut routing = match r.dest_routing.as_deref() {
            None | Some("keep") => source_routing.clone(),
            Some("discard") => None,
            Some(v) => Some(v.strip_prefix('=').unwrap_or(v).to_string()),
        };
        let mut version = match r.dest_version_type {
            Some(VersionType::External) | Some(VersionType::ExternalGte) => Some(source_version),
            _ => None,
        };
        // ES `Reindexer.buildRequest`: a hit without `_source` (disabled in
        // the source mapping, `"_source": false` in the request, stored
        // fields only) fails the request before this batch is written;
        // it is never indexed as `{}`.
        let raw = raw_sources.get(pos).and_then(|s| s.clone()).filter(|s| s.trim() != "null");
        let parsed = h.get("_source").filter(|s| !s.is_null()).map(|s| s.to_string());
        let Some(mut source) = raw.or(parsed) else {
            return Err(EsError::illegal_argument(format!("[{}][{hid}] didn't store _source", h["_index"].as_str().unwrap_or_default())));
        };
        let mut delete = false;
        if let Some(sc) = &script {
            let hit_index = h["_index"].as_str().unwrap_or_default().to_string();
            let meta = DocMeta { index: hit_index.clone(), id: Some(hid.to_string()), routing: source_routing.clone(), version: Some(source_version), ..Default::default() };
            let (o,transformed) = run_doc_script_source(sc, &source, &meta, "index")?;
            match o.op {
                DocOp::Noop => {
                    st.noops += 1;
                    continue;
                }
                DocOp::Delete => delete = true,
                _ => {}
            }
            source = transformed;
            if o.meta.index != hit_index {
                index = o.meta.index;
            }
            if o.meta.id.as_deref() != Some(hid) {
                id = o.meta.id;
            }
            if o.meta.routing != source_routing {
                routing = o.meta.routing;
            }
            if let Some(v) = o.meta.version.filter(|v| *v != source_version) {
                version = Some(v);
            }
        }
        let mut a = json!({"_index": index});
        if let Some(id) = &id {
            a["_id"] = json!(id);
        }
        if let Some(rt) = &routing {
            a["routing"] = json!(rt);
        }
        if delete {
            // ES deletes with the request's version under internal
            // (equality) semantics: an older or newer destination conflicts.
            if let Some(v) = version {
                a[REINDEX_VERSION_EQ] = json!(v);
            }
            out.push(Action { index, id: id.unwrap_or_default(), action: json!({"delete": a}), source: None });
            continue;
        }
        if let Some(v) = version {
            a["version"] = json!(v);
            a["version_type"] = json!(if r.dest_version_type == Some(VersionType::ExternalGte) { "external_gte" } else { "external" });
        }
        if let Some(p) = &r.pipeline {
            a["pipeline"] = json!(p);
        }
        let op = if r.dest_op_create { "create" } else { "index" };
        out.push(Action { index, id: id.unwrap_or_default(), action: json!({ op: a }), source: Some(source) });
    }
    Ok(out)
}

/// Count bulk item results into `st`; `true` when the reindex must stop
/// (a failure, or a version conflict without `conflicts: proceed`).
pub(crate) fn tally(r: &ByQueryRequest, actions: &[Action], items: &[Value], st: &mut Stats) -> bool {
    let mut abort = false;
    for (i, it) in items.iter().enumerate() {
        let Some(item) = it.as_object().and_then(|o| o.values().next()) else { continue };
        let status = item["status"].as_u64().unwrap_or(0) as u16;
        if (200..300).contains(&status) {
            match item["result"].as_str().unwrap_or("") {
                "created" => st.created += 1,
                "updated" => st.updated += 1,
                "deleted" => st.deleted += 1,
                "noop" => st.noops += 1,
                // A delete of a missing document is still "a delete".
                "not_found" => st.deleted += 1,
                _ => {}
            }
            continue;
        }
        let (index, id) = match actions.get(i) {
            Some(a) => (item["_index"].as_str().unwrap_or(&a.index).to_string(), a.id.clone()),
            None => (item["_index"].as_str().unwrap_or_default().to_string(), item["_id"].as_str().unwrap_or_default().to_string()),
        };
        let mut cause = item["error"].clone();
        if let Some(o) = cause.as_object_mut() {
            o.remove("root_cause");
        }
        let failure = json!({"index": index, "id": id, "cause": cause, "status": status});
        if status == 409 {
            st.version_conflicts += 1;
            if !r.conflicts_proceed {
                st.failures.push((status, failure));
                abort = true;
            }
        } else {
            st.failures.push((status, failure));
            abort = true;
        }
    }
    abort
}

#[cfg(test)]
mod tests {
    #[test]
    fn source_reads_keep_the_expression_and_its_options() {
        use elasticsearch_engine::node::ResolveOpts;
        assert_eq!(super::source_expression("al ias,idx-*"), "al%20ias,idx-%2A");
        assert_eq!(super::source_expression(""), "_all");
        assert_eq!(super::resolve_params(&ResolveOpts::default()), "ignore_unavailable=false&allow_no_indices=true&expand_wildcards=open");
        let o = ResolveOpts { ignore_unavailable: true, allow_no_indices: false, expand_open: false, expand_closed: false, expand_hidden: false };
        assert_eq!(super::resolve_params(&o), "ignore_unavailable=true&allow_no_indices=false&expand_wildcards=none");
    }

    #[test]
    fn response_reports_timeouts_and_search_failures() {
        let mut st = elasticsearch_engine::reindex::Stats::default();
        st.timed_out = true;
        let (status, v) = super::response(&st, -1.0, 1);
        assert_eq!((status, v["timed_out"].as_bool()), (408, Some(true)));
        st.timed_out = false;
        st.failures.push((503, serde_json::json!({"index": "s", "shard": 1, "status": 503, "reason": {"type": "no_shard_available_action_exception"}})));
        let (status, v) = super::response(&st, -1.0, 1);
        assert_eq!((status, v["failures"].as_array().map(Vec::len)), (503, Some(1)));
    }

    #[test]
    fn credentials_carry_the_system_identity() {
        let req = RestReq::try_new("POST", "/_bulk", &[
            ("authorization".into(), "Basic x".into()), ("x-rustic-internal-system".into(), "true".into()), ("content-type".into(), "application/json".into())], &[], 0, Default::default(), None).unwrap();
        let names: Vec<String> = credentials(&req).unwrap().iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(names, vec!["authorization", "x-rustic-internal-system"]);
    }

    use super::*;

    fn copy(index: &str, primary: bool, node: &str) -> elasticsearch_cluster::state::ShardRouting {
        elasticsearch_cluster::state::ShardRouting { index: index.into(), shard: 0, primary, state: elasticsearch_cluster::state::ShardState::Started,
            node_id: Some(node.into()), relocating_node_id: None, allocation_id: Some(elasticsearch_cluster::state::AllocationId { id: format!("{index}-{node}"), relocation_id: None }),
            recovery_source: None, unassigned_info: None, expected_shard_size_in_bytes: None }
    }

    fn state(tables: Vec<(&str, Vec<elasticsearch_cluster::state::ShardRouting>)>) -> elasticsearch_cluster::state::ClusterState {
        let mut s = elasticsearch_cluster::state::ClusterState::empty("t");
        for (index, copies) in tables {
            s.routing_table.indices.insert(index.into(), elasticsearch_cluster::state::IndexRoutingTable { index_uuid: "u".into(),
                shards: vec![elasticsearch_cluster::state::IndexShardRoutingTable { shard: 0, copies }] });
        }
        s
    }

    #[test]
    fn local_reindex_only_when_the_source_is_on_the_destination_primary() {
        let src: BTreeSet<String> = ["src".to_string()].into();
        // Source and destination primaries on different nodes: the engine
        // path would read nothing there.
        assert!(!colocated(&state(vec![("src", vec![copy("src", true, "n1")]), ("dst", vec![copy("dst", true, "n2")])]), &src, "dst"));
        // A replica of the source next to the destination primary will do.
        assert!(colocated(&state(vec![("src", vec![copy("src", true, "n1"), copy("src", false, "n2")]), ("dst", vec![copy("dst", true, "n2")])]), &src, "dst"));
        assert!(colocated(&state(vec![("src", vec![copy("src", true, "n2")]), ("dst", vec![copy("dst", true, "n2")])]), &src, "dst"));
        // A destination the reindex will create has no primary yet.
        assert!(!colocated(&state(vec![("src", vec![copy("src", true, "n1")])]), &src, "new"));
    }
    use std::collections::HashMap;

    fn req(body: Value) -> ByQueryRequest {
        parse_reindex(&body, &body.to_string(), &HashMap::new()).unwrap()
    }

    fn hit(id: &str, routing: Option<&str>, version: u64) -> Value {
        let mut h = json!({"_index": "src", "_id": id, "_version": version, "_source": {"f": id}});
        if let Some(r) = routing {
            h["_routing"] = json!(r);
        }
        h
    }

    #[test]
    fn actions_keep_routing_and_external_version() {
        let r = req(json!({"source": {"index": "src"}, "dest": {"index": "dst", "version_type": "external"}}));
        let mut st = Stats::default();
        let a = build_actions(&r, "dst", &[hit("1", Some("r1"), 3), hit("2", None, 1)], Path::new("/nonexistent"), &mut st).unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].action, json!({"index": {"_index": "dst", "_id": "1", "routing": "r1", "version": 3, "version_type": "external"}}));
        assert_eq!(a[1].action["index"].get("routing"), None);
        assert_eq!(serde_json::from_str::<Value>(a[0].source.as_ref().unwrap()).unwrap(),json!({"f":"1"}));
    }

    #[test]
    fn actions_routing_discard_override_and_create() {
        let r = req(json!({"source": {"index": "src"}, "dest": {"index": "dst", "routing": "=x", "op_type": "create"}}));
        let mut st = Stats::default();
        let a = build_actions(&r, "dst", &[hit("1", Some("r1"), 1)], Path::new("/nonexistent"), &mut st).unwrap();
        assert_eq!(a[0].action, json!({"create": {"_index": "dst", "_id": "1", "routing": "x"}}));
        let r = req(json!({"source": {"index": "src"}, "dest": {"index": "dst", "routing": "discard"}}));
        let a = build_actions(&r, "dst", &[hit("1", Some("r1"), 1)], Path::new("/nonexistent"), &mut st).unwrap();
        assert_eq!(a[0].action, json!({"index": {"_index": "dst", "_id": "1"}}));
    }

    /// Production review #191, #199, #241: the destination request is
    /// built first, then the script's metadata changes apply over it. A
    /// script clearing `_id` gets a generated id (no `_id` in the action);
    /// a script changing `_routing` wins over `dest.routing` (`discard` and
    /// `=constant`), an unchanged one keeps the policy; a scripted delete
    /// keeps the version condition.
    #[test]
    fn script_metadata_applies_over_the_destination_request() {
        let script = |s: &str| json!({"source": s});
        let mut st = Stats::default();
        let r = req(json!({"source": {"index": "src"}, "dest": {"index": "dst"}, "script": script("ctx._id = null")}));
        let a = build_actions(&r, "dst", &[hit("1", None, 1), hit("1", Some("other"), 1)], Path::new("/nonexistent"), &mut st).unwrap();
        assert_eq!(a.len(), 2);
        for x in &a {
            assert_eq!(x.action["index"].get("_id"), None, "a cleared id is generated: {}", x.action);
        }
        for policy in ["discard", "=fixed"] {
            let r = req(json!({"source": {"index": "src"}, "dest": {"index": "dst", "routing": policy}, "script": script("ctx._routing = ctx._source.f + '-new'")}));
            let a = build_actions(&r, "dst", &[hit("a", Some("old-a"), 1), hit("b", Some("old-b"), 1)], Path::new("/nonexistent"), &mut st).unwrap();
            assert_eq!(a[0].action["index"]["routing"], json!("a-new"), "{policy}: {}", a[0].action);
            assert_eq!(a[1].action["index"]["routing"], json!("b-new"), "{policy}: {}", a[1].action);
            // A script leaving the routing alone keeps the policy's.
            let r = req(json!({"source": {"index": "src"}, "dest": {"index": "dst", "routing": policy}, "script": script("ctx._source.g = 1")}));
            let a = build_actions(&r, "dst", &[hit("a", Some("old-a"), 1)], Path::new("/nonexistent"), &mut st).unwrap();
            let want = if policy == "discard" { None } else { Some(json!("fixed")) };
            assert_eq!(a[0].action["index"].get("routing").cloned(), want, "{policy}");
        }
        let r = req(json!({"source": {"index": "src"}, "dest": {"index": "dst", "version_type": "external"}, "script": script("ctx.op = 'delete'")}));
        let a = build_actions(&r, "dst", &[hit("1", Some("r1"), 3)], Path::new("/nonexistent"), &mut st).unwrap();
        assert_eq!(a[0].action, json!({"delete": {"_index": "dst", "_id": "1", "routing": "r1", REINDEX_VERSION_EQ: 3}}));
        let r = req(json!({"source": {"index": "src"}, "dest": {"index": "dst", "version_type": "external"}, "script": script("ctx._version = 7; ctx.op = 'delete'")}));
        let a = build_actions(&r, "dst", &[hit("1", None, 3)], Path::new("/nonexistent"), &mut st).unwrap();
        assert_eq!(a[0].action["delete"][REINDEX_VERSION_EQ], json!(7));
        // Internal destination versioning: no version condition (ES MATCH_ANY).
        let r = req(json!({"source": {"index": "src"}, "dest": {"index": "dst"}, "script": script("ctx.op = 'delete'")}));
        let a = build_actions(&r, "dst", &[hit("1", None, 3)], Path::new("/nonexistent"), &mut st).unwrap();
        assert_eq!(a[0].action, json!({"delete": {"_index": "dst", "_id": "1"}}));
    }

    /// ES BulkByScroll retries on the cluster path: a whole-response 429
    /// and per-item 429s are re-sent (only the rejected items), counted in
    /// retries.bulk, and succeed once accepted.
    #[tokio::test]
    async fn rejected_bulks_and_items_are_retried() {
        let r = req(json!({"source": {"index": "src"}, "dest": {"index": "dst"}}));
        let mut st = Stats::default();
        let acts = build_actions(&r, "dst", &[hit("1", None, 1), hit("2", None, 1)], Path::new("/nonexistent"), &mut st).unwrap();
        let mut calls = 0;
        let mut retries = 0;
        let mut sent = Vec::new();
        let items = bulk_retrying(&acts, &mut retries, |nd| {
            calls += 1;
            sent.push(std::str::from_utf8(&nd).unwrap().lines().count());
            let call = calls;
            async move {
                match call {
                    1 => RestResp::error(429, "es_rejected_execution_exception", "queue full"),
                    2 => RestResp::json(200, &json!({"errors": true, "items": [
                        {"index": {"_index": "dst", "_id": "1", "status": 201, "result": "created"}},
                        {"index": {"_index": "dst", "_id": "2", "status": 429, "error": {"type": "es_rejected_execution_exception", "reason": "busy"}}}]})),
                    _ => RestResp::json(200, &json!({"errors": false, "items": [
                        {"index": {"_index": "dst", "_id": "2", "status": 201, "result": "created"}}]})),
                }
            }
        }).await.expect("retried to success");
        assert_eq!(calls, 3);
        assert_eq!(retries, 2);
        assert_eq!(sent, vec![4, 4, 2], "only the rejected item is re-sent");
        assert!(!tally(&r, &acts, &items, &mut st));
        assert_eq!(st.created, 2);
        assert!(st.failures.is_empty());
    }

    #[test]
    fn tally_counts_and_aborts_on_conflict() {
        let r = req(json!({"source": {"index": "src"}, "dest": {"index": "dst"}}));
        let mut st = Stats::default();
        let acts = build_actions(&r, "dst", &[hit("1", None, 1), hit("2", None, 1), hit("3", None, 1)], Path::new("/nonexistent"), &mut st).unwrap();
        let items = vec![
            json!({"index": {"_index": "dst", "_id": "1", "status": 201, "result": "created"}}),
            json!({"index": {"_index": "dst", "_id": "2", "status": 200, "result": "updated"}}),
            json!({"index": {"_index": "dst", "_id": "3", "status": 409, "error": {"type": "version_conflict_engine_exception", "root_cause": [], "reason": "x"}}}),
        ];
        assert!(tally(&r, &acts, &items, &mut st));
        assert_eq!((st.created, st.updated, st.version_conflicts), (1, 1, 1));
        assert_eq!(st.failures.len(), 1);
        assert_eq!(st.failures[0].1, json!({"index": "dst", "id": "3", "cause": {"type": "version_conflict_engine_exception", "reason": "x"}, "status": 409}));
        let r = req(json!({"source": {"index": "src"}, "dest": {"index": "dst"}, "conflicts": "proceed"}));
        let mut st = Stats::default();
        assert!(!tally(&r, &acts, &items, &mut st));
        assert_eq!((st.version_conflicts, st.failures.len()), (1, 0));
    }

    /// A hit without `_source` (absent or null) fails the batch with ES's
    /// error; it is never written as `{}`.
    #[test]
    fn a_hit_without_source_fails_the_batch() {
        let r = req(json!({"source": {"index": "src"}, "dest": {"index": "dst"}}));
        for h in [json!({"_index": "src", "_id": "x", "_version": 1}), json!({"_index": "src", "_id": "x", "_version": 1, "_source": null})] {
            let mut st = Stats::default();
            let e = build_actions(&r, "dst", &[hit("1", None, 1), h.clone()], Path::new("/nonexistent"), &mut st).unwrap_err();
            assert_eq!(e.kind, "illegal_argument_exception", "{h}");
            assert_eq!(e.reason, "[src][x] didn't store _source", "{h}");
            let raw = vec![Some("{\"f\":1}".to_string()), None];
            let e = build_actions_source(&r, "dst", &[hit("1", None, 1), h.clone()], &raw, Path::new("/nonexistent"), &mut st).unwrap_err();
            assert_eq!(e.reason, "[src][x] didn't store _source", "{h}");
        }
        let mut st = Stats::default();
        let empty = json!({"_index": "src", "_id": "e", "_version": 1, "_source": {}});
        assert_eq!(build_actions(&r, "dst", &[empty], Path::new("/nonexistent"), &mut st).unwrap().len(), 1, "an empty source is copied");
    }
}
