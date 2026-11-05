//! `_search`, scroll, point in time, `_count`, `_msearch`.

use crate::{blocking, method_not_allowed, Req, Resp, R};
use crate::search_task::Kind;
use axum::http::Method;
use elasticsearch_engine::error::{EsError, Result};
use elasticsearch_engine::highlight::source_values;
use elasticsearch_engine::node::{Node, PitCtx, ResolveOpts, ScrollCtx};
use elasticsearch_engine::search::{self, ExecOpts, SearchRequest, SortSpec, Target};
use crate::security::access::ReadAccess;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[path = "search_validate.rs"]
pub(crate) mod search_validate;

pub async fn handle(node: Arc<Node>, req: Req, index: Option<String>) -> R {
    let off = if index.is_some() { 1 } else { 0 };
    let api = req.seg(off).to_string();
    let sub = req.seg(off + 1).to_string();
    let m = req.method.clone();
    match (api.as_str(), sub.as_str(), &m) {
        ("_search", "", &Method::GET) | ("_search", "", &Method::POST) => {
            let task = begin_task(&node, &req, Kind::Search, index.as_deref())?;
            as_task(task, move || search(&node, &req, index.as_deref())).await
        }
        ("_search", "scroll", &Method::GET) | ("_search", "scroll", &Method::POST) if index.is_none() => {
            let task = begin_task(&node, &req, Kind::Scroll, None)?;
            as_task(task, move || scroll(&node, &req)).await
        }
        ("_search", "scroll", &Method::DELETE) if index.is_none() => blocking(move || clear_scroll(&node, &req)).await,
        ("_count", "", &Method::GET) | ("_count", "", &Method::POST) => {
            let task = begin_task(&node, &req, Kind::Search, index.as_deref())?;
            as_task(task, move || count(&node, &req, index.as_deref())).await
        }
        ("_msearch", "", &Method::GET) | ("_msearch", "", &Method::POST) => blocking(move || msearch(&node, &req, index.as_deref())).await,
        ("_pit", "", &Method::POST) if index.is_some() => blocking(move || open_pit(&node, &req, index.as_deref().unwrap())).await,
        ("_pit", "", &Method::DELETE) if index.is_none() => blocking(move || close_pit(&node, &req)).await,
        ("_knn_search", "", &Method::GET) | ("_knn_search", "", &Method::POST) if index.is_some() => {
            let task = begin_task(&node, &req, Kind::Search, index.as_deref())?;
            as_task(task, move || knn_search(&node, &req, index.as_deref().unwrap())).await
        }
        ("_search", "", _) | ("_count", "", _) | ("_msearch", "", _) | ("_knn_search", "", _) => method_not_allowed(&req, &["GET", "POST"]),
        _ => crate::not_found_handler(&req),
    }
}

fn begin_task(node: &Arc<Node>, req: &Req, kind: Kind, index: Option<&str>) -> Result<Option<crate::search_task::Guard>> {
    // Task metadata refused by the breaker: the search runs untracked (as
    // before tasks), not failed for its bookkeeping.
    Ok(crate::search_task::begin(node, req, kind, index).unwrap_or(None))
}

/// Run a search route's work as its search task ([`crate::search_task`]):
/// on the executor under the task's cancellation token. The task ends with
/// the handler; a handler dropped before it answers cancels it first.
async fn as_task<F>(guard: Option<crate::search_task::Guard>, f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
{
    let token = guard.as_ref().map(|g| g.token());
    // Dropped with this future before the answer: the work is cancelled.
    let mut on_drop = token.clone().map(crate::search_task::CancelOnDrop::new);
    // The task lists until the executor thread is done with its work.
    let out = blocking(move || {
        let mut guard = guard;
        // Cancelled while queued (its client went away, its parent was
        // cancelled or banned): the thread is not spent on it.
        if token.as_ref().is_some_and(|t| t.is_cancelled()) {
            return elasticsearch_engine::rustic_core::cancel::with_search(token, || Err(elasticsearch_engine::search::task_cancelled()));
        }
        let out = elasticsearch_engine::rustic_core::cancel::with_search(token, f);
        if let Some(guard) = guard.as_mut() {
            guard.complete();
        }
        out
    }).await;
    if let Some(on_drop) = on_drop.as_mut() {
        on_drop.disarm();
    }
    out
}

fn keep_alive(s: Option<&str>) -> Result<Duration> {
    keep_alive_named(s, "keep_alive")
}

fn keep_alive_named(s: Option<&str>, name: &str) -> Result<Duration> {
    let s = s.unwrap_or("5m");
    let ms = elasticsearch_engine::date::parse_time_value(s)
        .ok_or_else(|| EsError::illegal_argument(format!("failed to parse setting [{name}] with value [{s}] as a time value: unit is missing or unrecognized")))?;
    Ok(Duration::from_millis(ms.max(0) as u64))
}

static CTX_SEQ: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(1);

/// An ES-byte-compatible scroll id for a new search context on this node.
fn new_ctx_id() -> String {
    let n = CTX_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    elasticsearch_engine::search_ids::new_scroll_id(&uuid_like(), n, elasticsearch_engine::node::NODE_ID)
}

/// An ES-byte-compatible point-in-time id over `targets`.
fn new_pit_id(targets: &[Target]) -> String {
    let session = uuid_like();
    let shards: Vec<(String, String, u32, String, i64, String)> = targets
        .iter()
        .map(|t| {
            let n = CTX_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (t.svc.name.clone(), t.svc.uuid.clone(), 0, session.clone(), n, elasticsearch_engine::node::NODE_ID.to_string())
        })
        .collect();
    elasticsearch_engine::search_ids::new_pit_id(&shards)
}

fn uuid_like() -> String {
    elasticsearch_engine::index::new_uuid()
}

/// ES `RestSearchScrollAction` / `RestClearScrollAction` body parsing: only
/// `scroll_id` (string, or array for clear) and `scroll` are accepted.
fn scroll_body(req: &Req, allow_array: bool) -> Result<Value> {
    let body = req.json().map_err(|e| EsError::illegal_argument("Failed to parse request body").caused_by(e))?;
    if let Some(o) = body.as_object() {
        for (k, v) in o {
            let ok = match k.as_str() {
                "scroll_id" if allow_array => !v.is_object(),
                "scroll_id" => v.is_string(),
                "scroll" => v.is_string() && !allow_array,
                _ => false,
            };
            if !ok {
                return Err(EsError::illegal_argument(format!("Unknown parameter [{k}] in request body or parameter is of the wrong type[{}] ", elasticsearch_engine::jsonpos::token_name(v))));
            }
        }
    }
    Ok(body)
}

/// Terms-lookup through realtime get. With security on, the lookup is a read
/// of the lookup index: it needs the read privilege and honours its DLS/FLS.
fn terms_lookup<'a>(node: &'a Node, access: Option<&'a ReadAccess>) -> impl Fn(&str, &str, &str, Option<&str>) -> Result<Vec<Value>> + Sync + 'a {
    move |idx: &str, id: &str, path: &str, routing: Option<&str>| {
        let svc = node.write_index(idx, false)?;
        let acc = match access {
            Some(a) => a.lookup_access(&svc.name, idx)?,
            None => None,
        };
        if let Some(acc) = &acc {
            if !search::doc_visible(&svc, id, acc, false)? {
                return Ok(vec![]);
            }
        }
        // The lookup document lives on the shard its routing (or id) selects.
        let Some(g) = (match routing {
            Some(r) => svc.get_routed(id, Some(r), true)?,
            None => svc.get(id)?,
        }) else {
            return Ok(vec![]);
        };
        let mut src: Value = g.source.as_deref().and_then(|s| serde_json::from_str(s).ok()).unwrap_or(Value::Null);
        if let Some(fa) = acc.as_ref().and_then(|a| a.fields.as_ref()) {
            src = fa.filter_source(&src);
        }
        Ok(source_values(&src, path).into_iter().cloned().collect())
    }
}

/// Stamp the resolved targets with the caller's DLS/FLS (default-deny on failure).
fn restrict(access: Option<&ReadAccess>, targets: Vec<Target>) -> Result<Vec<Target>> {
    match access {
        Some(a) => a.apply(targets),
        None => Ok(targets),
    }
}

fn targets_for(node: &Node, req: &Req, index: Option<&str>) -> Result<Vec<Target>> {
    let mut opts = ResolveOpts::from_params(&req.params);
    if index.is_none() || index == Some("_all") || index == Some("*") {
        opts.allow_no_indices = true;
    }
    // `_count` accepts cross-cluster expressions: plan the remote parts.
    Ok(elasticsearch_engine::ccr::remote::resolve_local_part(node, index.unwrap_or("_all"), opts)?.0)
}

/// Restrict targets to the shards the request's `routing` values (and
/// the aliases' `search_routing`, intersected) route to.
pub fn apply_search_routing(targets: Vec<Target>, routing: Option<&str>) -> Vec<Target> {
    let vals: Vec<String> = routing.map(|r| r.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default();
    if vals.is_empty() {
        return targets;
    }
    targets
        .into_iter()
        .filter_map(|mut t| {
            let merged: Vec<String> = match &t.routing {
                None => vals.clone(),
                Some(a) => {
                    let alias: Vec<&str> = a.split(',').map(str::trim).collect();
                    vals.iter().filter(|v| alias.contains(&v.as_str())).cloned().collect()
                }
            };
            if merged.is_empty() {
                return None;
            }
            t.routing = Some(merged.join(","));
            Some(t)
        })
        .collect()
}

fn params_with(req: &Req) -> HashMap<String, String> {
    req.params.clone()
}

/// A request-level check that ES only hits while creating the shard query
/// (`search_phase_execution_exception` whose `failed_shards` name no index)
/// fails on every targeted index: resolve the targets (an unknown index wins)
/// and report one grouped shard failure per index, as ES does.
fn attach_shard_index(node: &Node, e: EsError, index: Option<&str>, resolve: ResolveOpts, access: Option<&ReadAccess>) -> EsError {
    let unnamed = e.kind == "search_phase_execution_exception"
        && e.extra.iter().any(|(k, v)| k == "failed_shards" && v.as_array().is_some_and(|a| a.len() == 1 && a[0].get("index").and_then(Value::as_str) == Some("")));
    if !unnamed {
        return e;
    }
    let mut ropts = resolve;
    if index.is_none() {
        ropts.allow_no_indices = true;
    }
    let targets = match node.resolve(index.unwrap_or("_all"), ropts).and_then(|t| restrict(access, t)) {
        Ok(t) => t,
        Err(re) => return re,
    };
    let mut seen = std::collections::BTreeSet::new();
    let mut e = e;
    for (k, v) in e.extra.iter_mut() {
        if k != "failed_shards" {
            continue;
        }
        let tmpl = v[0].clone();
        let mut out = Vec::new();
        for t in &targets {
            if !seen.insert(t.svc.name.clone()) {
                continue;
            }
            let mut f = tmpl.clone();
            f["index"] = json!(t.svc.name);
            if let Some(r) = f.get_mut("reason").and_then(Value::as_object_mut) {
                if r.contains_key("index") {
                    r.insert("index_uuid".into(), json!(t.svc.uuid));
                    r.insert("index".into(), json!(t.svc.name));
                }
            }
            out.push(f);
        }
        if !out.is_empty() {
            *v = Value::Array(out);
        }
    }
    e
}

/// Shared by `_search` and `_msearch`.
/// Percent-encode a URI component, keeping index-expression characters.
pub(crate) fn uri_component(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~*,".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// The `_search` URI an internal search sends through the cluster coordinator.
pub(crate) fn coordinator_search_uri(index: &str, params: &HashMap<String, String>) -> String {
    let mut query: Vec<String> = params.iter().map(|(k, v)| format!("{}={}", uri_component(k), uri_component(v))).collect();
    query.sort();
    format!("/{}/_search{}", uri_component(index), if query.is_empty() { String::new() } else { format!("?{}", query.join("&")) })
}

/// A search an API runs internally (rollup search, graph explore). On a
/// cluster it goes through the coordinator (every shard, from a started
/// copy, as the caller); the local engine alone holds only this node's
/// copies.
pub(crate) fn coordinated_search(node: &Node, req: &Req, body: &Value, params: &HashMap<String, String>, index: &str, opts: ResolveOpts, access: Option<&ReadAccess>) -> Result<String> {
    match req.extensions.get::<crate::ClusterRest>() {
        Some(rest) => (rest.0)("POST", &coordinator_search_uri(index, params), Some(body.clone())).map(|v| v.to_string()),
        None => run_search(node, body, params, Some(index), opts, access),
    }
}

/// Is this a whole-index read of this engine by a request (a client's, or
/// the cluster's), which must not answer for some of a sparse index's
/// shards only ([`search::require_local_shards`])? A cluster shard-level
/// read names its shard and never reaches the check whole; `_mvt` and
/// `_fleet_search` coordinate their searches through the cluster. Only the
/// rank plan phase (metadata, checked by the caller) and API-internal
/// searches with no request of their own read what is local.
fn whole_local_read(ext: Option<&axum::http::Extensions>) -> bool {
    ext.is_some()
}

pub(crate) fn run_search(node: &Node, body: &Value, params: &HashMap<String, String>, index: Option<&str>, resolve: ResolveOpts, access: Option<&ReadAccess>) -> Result<String> {
    run_search_ext(node, body, params, index, resolve, access, None)
}

/// [`run_search`] with the http request extensions of an in-process caller
/// (supplied [`crate::DfsStats`], [`crate::DfsPhaseOnly`]).
/// Enter the test modules the node runs with (query parsers they register).
pub(crate) fn test_modules(node: &Node) -> elasticsearch_engine::warnings::TestModulesGuard {
    let modules: Vec<String> = node
        .node_settings
        .read()
        .get(elasticsearch_engine::warnings::TEST_MODULES_SETTING)
        .and_then(Value::as_str)
        .map(|s| s.split(',').filter(|m| !m.is_empty()).map(str::to_string).collect())
        .unwrap_or_default();
    elasticsearch_engine::warnings::enter_test_modules(modules)
}

/// A `Warning` response header (`299 Elasticsearch-<version>-<hash> "msg"`).
pub(crate) fn warning_header(msg: &str) -> (&'static str, String) {
    ("Warning", format!("299 Elasticsearch-{}-{} \"{msg}\"", elasticsearch_engine::node::ES_VERSION, crate::cluster::BUILD_HASH))
}

fn run_search_ext(node: &Node, body: &Value, params: &HashMap<String, String>, index: Option<&str>, resolve: ResolveOpts, access: Option<&ReadAccess>, ext: Option<&axum::http::Extensions>) -> Result<String> {
    let _modules = test_modules(node);
    if let Some(expr) = index.filter(|e| crate::ccs::has_remote(e)) {
        // Validate the source before resolving remote index expressions.
        search::parse_request(body, params)?;
        let plan = crate::ccs::resolve(node, expr, resolve)?;
        let start = Instant::now();
        let result = run_search_inner(node, body, params, index, resolve, access, ext, Some(&plan));
        node.ccs_stats.lock().record(start.elapsed().as_millis() as u64, &plan.aliases, &plan.skipped, result.is_ok(), params.get("ccs_minimize_roundtrips").is_none_or(|v| v != "false"), expr.contains(['*', '?']));
        return crate::ccs::finish(result?, &plan);
    }
    run_search_inner(node, body, params, index, resolve, access, ext, None)
}

fn run_search_inner(node: &Node, body: &Value, params: &HashMap<String, String>, index: Option<&str>, resolve: ResolveOpts, access: Option<&ReadAccess>, ext: Option<&axum::http::Extensions>, ccs: Option<&crate::ccs::Plan>) -> Result<String> {
    for b in ["typed_keys", "track_scores", "explain", "version", "seq_no_primary_term", "rest_total_hits_as_int", "allow_partial_search_results", "ccs_minimize_roundtrips", "request_cache"] {
        if let Some(v) = params.get(b) {
            if !matches!(v.as_str(), "true" | "false" | "") {
                return Err(EsError::illegal_argument(format!("Failed to parse value [{v}] as only [true] or [false] are allowed.")));
            }
        }
    }
    let start = Instant::now();
    // A coordinator's rank plan travels in the body (only from the coordinator).
    let rank_mode = ext.and_then(|e| e.get::<crate::InternalRankPlan>()).copied();
    let plan_key = elasticsearch_engine::retriever::RANK_PLAN_KEY;
    let mut plan_in = None;
    let stripped;
    let body = match (rank_mode, body.get(plan_key)) {
        (Some(_), Some(p)) => {
            plan_in = Some(elasticsearch_engine::retriever::RankPlan::from_json(p)?);
            let mut b = body.clone();
            if let Some(o) = b.as_object_mut() {
                o.remove(plan_key);
            }
            stripped = b;
            &stripped
        }
        _ => body,
    };
    // A query_then_fetch fetch phase names its context and docs in the body.
    let qtf = ext.and_then(|e| e.get::<crate::InternalQtf>()).copied();
    let mut fetch_in: Option<(String, Vec<(usize, usize, u32, f32)>)> = None;
    let fetch_stripped;
    let body = match (qtf, body.get(crate::QTF_FETCH_KEY)) {
        (Some(crate::InternalQtf::Fetch), Some(f)) => {
            fetch_in = Some(qtf_fetch_spec(f).ok_or_else(|| EsError::illegal_argument("malformed query_then_fetch fetch request"))?);
            let mut b = body.clone();
            if let Some(o) = b.as_object_mut() {
                o.remove(crate::QTF_FETCH_KEY);
            }
            fetch_stripped = b;
            &fetch_stripped
        }
        _ => body,
    };
    let orig_body = body;
    // Structural validation first (ES parses the whole source before any rewrite).
    let parse_source = |body: &Value| {
        let mut environment = elasticsearch_engine::script::ScriptEnv::new(None, None);
        environment.raw_search_scripts = ext.and_then(|e| e.get::<elasticsearch_engine::source_json::Source>())
            .map(elasticsearch_engine::source_json::search_scripts).unwrap_or_default();
        elasticsearch_engine::script::with_env(environment, || search::parse_request(body, params))
    };
    let parsed = parse_source(body)
        .map_err(|e| attach_shard_index(node, e, index, resolve, access))?;
    // Query rules (`rule` query) rewrite.
    let pre = elasticsearch_engine::search_ext::preprocess_search_body(node, body)?;
    let body = pre.as_ref().unwrap_or(body);
    let mut req = if pre.is_some() {
        parse_source(body)?
    } else {
        parsed
    };
    req.raw_script_fields = ext.and_then(|e|e.get::<elasticsearch_engine::source_json::Source>()).and_then(|s|s.get("script_fields")).cloned();
    req.raw_runtime_mappings = ext.and_then(|e|e.get::<elasticsearch_engine::source_json::Source>()).and_then(|s|s.get("runtime_mappings")).cloned();
    req.raw_search_scripts = ext.and_then(|e|e.get::<elasticsearch_engine::source_json::Source>()).map(elasticsearch_engine::source_json::search_scripts).unwrap_or_default();
    req.raw_sort = ext.and_then(|e|e.get::<elasticsearch_engine::source_json::Source>()).and_then(|s|s.get("sort")).cloned();
    rank_license(node, body)?;
    if let Some(p) = plan_in {
        req.rank_plan = Some(Arc::new(p));
    }
    req.allow_expensive_queries = node.allow_expensive_queries();
    // ES `SearchService`: a request without its own `timeout` takes
    // `search.default_search_timeout` (`-1`, the default: none).
    if req.timeout.is_none() && body.get("timeout").is_none() && !params.contains_key("timeout") {
        req.timeout = node.default_search_timeout();
    }
    search_validate::validate_request(body, params, &req)?;
    if params.contains_key("scroll") {
        // Scroll snapshots retain an exact total on every page. Validate
        // explicit false/limited settings before normalizing the implicit
        // default, then store this exact-tracking request in ScrollCtx.
        req.track_total_hits = Some(elasticsearch_engine::rustic_core::query::TrackTotalHits::True);
    }
    req.index_boosts = search_validate::index_boosts(node, body, params)?;
    req.dfs_stats = ext.and_then(|e| e.get::<Arc<crate::DfsStats>>()).cloned();
    req.admitted_dfs_stats = ext.and_then(|e|e.get::<elasticsearch_engine::search::phases::AdmittedDfsStats>()).cloned();
    let dfs_only = ext.is_some_and(|e| e.get::<crate::DfsPhaseOnly>().is_some());
    if !dfs_only {
        crate::observability::record_aggs(node, index, body);
    }
    let lookup = terms_lookup(node, access);
    if let Some((ctx, docs)) = fetch_in {
        // The query phase's reader context, if this requester opened it for
        // this index and shard: freed by this fetch.
        let owner = fetch_owner(access, index);
        let Some(targets) = node.take_fetch_context(&ctx, &owner).map(|c| c.targets) else {
            if docs.is_empty() {
                // A free ahead of its query phase (or after it expired).
                node.free_unregistered_fetch_context(&ctx, &owner);
            }
            return Err(EsError::search_context_missing(&ctx));
        };
        if docs.is_empty() {
            return Ok("{}".to_string());
        }
        req.from = 0;
        req.size = docs.len();
        req.aggs = None;
        req.track_total_hits = Some(elasticsearch_engine::rustic_core::query::TrackTotalHits::False);
        req.fetch_docs = Some(Arc::new(docs));
        req.fetch_refs = ext.is_some_and(|e| e.get::<crate::InternalQtfLight>().is_some());
        let out = search::execute(&targets, &req, &ExecOpts { terms_lookup: Some(&lookup), cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None })?;
        if let Some((end, lens)) = &out.hit_spans {
            FETCH_HIT_SPANS.with(|s| *s.borrow_mut() = Some(crate::encode_hit_spans(out.json.len() - end, lens)));
        }
        return Ok(out.json);
    }
    // Point in time.
    if let Some(pit) = body.get("pit") {
        if index.is_some() {
            return Err(EsError::action_validation(&["[indices] cannot be used with point in time. Do not specify any index with point in time."]));
        }
        let id = pit.get("id").and_then(Value::as_str).ok_or_else(|| EsError::illegal_argument("point in time id is not provided"))?.to_string();
        elasticsearch_engine::search_ids::parse_pit_id(&id)?;
        let ka = pit.get("keep_alive").and_then(Value::as_str).map(|s| keep_alive(Some(s))).transpose()?;
        let (targets, lost) = {
            let mut pits = node.pits.lock();
            let ctx = pits.get_mut(&id).ok_or_else(|| EsError::search_context_missing(&id))?;
            if let Some(ka) = ka {
                ctx.keep_alive = ka;
            }
            ctx.expires = Instant::now() + ctx.keep_alive;
            (ctx.targets.clone(), ctx.lost.clone())
        };
        let targets = restrict(access, targets)?;
        if dfs_only {
            let stats = search::dfs_phase(&targets, &req, &ExecOpts { terms_lookup: Some(&lookup), cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None })?;
            return Ok(serde_json::to_string(&stats)?);
        }
        if !req.sort.is_empty() && !req.sort.iter().any(|s| s.field == "_shard_doc") {
            req.sort.push(SortSpec { field: "_shard_doc".into(), descending: false, missing: None, mode: None, unmapped_type: None, format: None, numeric_type: None });
            // A search_after without the implicit tiebreaker value skips every hit equal on the user's sort.
            if let Some(sa) = req.search_after.as_mut() {
                if sa.len() + 1 == req.sort.len() {
                    sa.push(json!(i64::MAX));
                }
            }
        }
        let out = search::execute(&targets, &req, &ExecOpts { terms_lookup: Some(&lookup), cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: Some(id) })?;
        return Ok(elasticsearch_engine::node::report_lost_targets(out.json, &lost));
    }
    let mut ropts = resolve;
    if index.is_none() {
        ropts.allow_no_indices = true;
    }
    let targets = restrict(access, match ccs { Some(plan) => plan.targets.clone(), None => node.resolve(index.unwrap_or("_all"), ropts)? })?;
    let targets = apply_search_routing(targets, params.get("routing").map(String::as_str));
    if rank_mode.is_none() && whole_local_read(ext) {
        search::require_local_shards(&targets)?;
    }
    if params.get("routing").is_some_and(|r| !r.is_empty()) {
        if let Some(t) = targets.iter().find(|t| t.svc.is_time_series()) {
            return Err(elasticsearch_engine::tsdb::mode_error("searching with a specified routing", &t.svc.name));
        }
    }
    crate::obs_search::check_read_blocks(&targets)?;
    search_validate::check_limits(body, &req, &targets, params.contains_key("scroll"))?;
    if !ext.is_some_and(|e| e.get::<crate::CoordinatorUsage>().is_some()) {
        crate::usage::record_analytics(node, body);
    }
    if let Some(aggs) = body.get("aggs").or_else(|| body.get("aggregations")).and_then(Value::as_object) {
        let fbf = {
            let cs = node.cluster_settings.read();
            let v = cs.1.get("search.aggs.rewrite_to_filter_by_filter").or(cs.0.get("search.aggs.rewrite_to_filter_by_filter")).cloned();
            v.map(|v| v.as_bool().unwrap_or_else(|| v.as_str() != Some("false"))).unwrap_or(true)
        };
        elasticsearch_engine::aggs::validate_order_paths(aggs, fbf)?;
    }
    if dfs_only {
        let stats = search::dfs_phase(&targets, &req, &ExecOpts { terms_lookup: Some(&lookup), cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None })?;
        return Ok(serde_json::to_string(&stats)?);
    }
    if rank_mode == Some(crate::InternalRankPlan::Plan) && (req.retriever.is_some() || req.rank.is_some()) {
        let rest = ext.and_then(|e| e.get::<crate::ClusterRest>()).cloned().ok_or_else(|| EsError::internal("rank plan phase without a coordinator"))?;
        let many = ext.and_then(|e| e.get::<crate::ClusterRestMany>()).cloned();
        let child = ClusterChildSearch { rest, many, uri: child_search_uri(index, params) };
        let opts = ExecOpts { terms_lookup: Some(&lookup), cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None };
        let out = elasticsearch_engine::retriever::plan(&targets, &req, body, &opts, &child)?;
        return Ok(json!({ "_rustic_final_body": out }).to_string());
    }
    if let Some(sc) = params.get("scroll") {
        let ka = keep_alive(Some(sc))?;
        search_validate::check_keep_alive(node, ka)?;
        if req.search_after.is_some() {
            return Err(EsError::action_validation(&["using [search_after] is not allowed in a scroll context"]));
        }
        search_validate::check_scroll_contexts(node, targets.len().max(1))?;
        let snap = node.snapshot(&targets)?;
        let id = new_ctx_id();
        let out = search::execute(&snap, &req, &ExecOpts { terms_lookup: Some(&lookup), cursors: None, max_buckets: node.max_buckets(), scroll_id: Some(id.clone()), pit_id: None })?;
        let mut next = req.clone();
        next.from = 0;
        next.aggs = None;
        // Shards that can_match skipped never opened a search context: when
        // every shard was skipped the scroll id holds none (ES 9).
        let skipped_all = out.json.contains("\"hits\":[]")
            && serde_json::from_str::<Value>(&out.json).ok().and_then(|v| {
                let sh = v.get("_shards")?;
                let (t, sk) = (sh.get("total")?.as_u64()?, sh.get("skipped")?.as_u64()?);
                Some(t > 0 && t == sk)
            }) == Some(true);
        let snap = if skipped_all { Vec::new() } else { snap };
        let accounting = Some(elasticsearch_engine::node::context_charge(&snap, out.cursors.len()));
        node.scrolls.lock().insert(id, ScrollCtx { targets: snap, req: next, cursors: out.cursors, keep_alive: ka, expires: Instant::now() + ka, accounting, lost: Vec::new() });
        return Ok(out.json);
    }
    if qtf == Some(crate::InternalQtf::Query) {
        // query_then_fetch query phase: search pinned views and keep them
        // for the fetch phase (ES registers a `ReaderContext`).
        // The whole request was parsed and checked (index limits above),
        // fetch-only parts included, as ES checks them when the shard
        // request creates its context: an error is the query phase's, not a
        // lost fetch. `search::execute` checks the script fields in its
        // script environment and renders none of those parts.
        let pinned = search::pin_views(&targets)?;
        let id = ext.and_then(|e| e.get::<crate::InternalQtfCtx>()).map(|c| c.0.clone()).unwrap_or_else(new_ctx_id);
        // Admit before scoring, without publishing an expiring/fetchable
        // context until the query is complete. Refusal must not require a
        // second query against a potentially different snapshot.
        let pinned_coordinator = ext.is_some_and(|e| e.get::<crate::InternalQtfPinned>().is_some());
        let admission = match node.reserve_fetch_context(&pinned) {
            Some(admission) => Some(admission),
            None if pinned_coordinator => return Err(EsError::new(429, "es_rejected_execution_exception", "unable to admit query_then_fetch reader context")),
            // An older coordinator re-runs a shard that kept no context in
            // one phase (as before pinned query_then_fetch): answer it so.
            None => None,
        };
        req.doc_refs = true;
        req.light_refs = ext.is_some_and(|e| e.get::<crate::InternalQtfLight>().is_some());
        let out = search::execute(&pinned, &req, &ExecOpts { terms_lookup: Some(&lookup), cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None })?;
        // Where the light query hits are (counted from the end, so the
        // context id written before them below moves none), for a
        // coordinator that reads a large page's hits as text.
        if req.light_refs {
            if let Some((end, lens)) = &out.hit_spans {
                FETCH_HIT_SPANS.with(|s| *s.borrow_mut() = Some(crate::encode_hit_spans(out.json.len() - end, lens)));
            }
        }
        crate::usage::record(&targets, body);
        let Some(admission) = admission else { return Ok(out.json) };
        // Empty results and error paths drop the capacity reservation.
        if out.hits == 0 { return Ok(out.json); }
        if !node.register_reserved_fetch_context(id.clone(), pinned, fetch_owner(access, index), admission) {
            return Err(EsError::new(429, "es_rejected_execution_exception", "query_then_fetch reader context cancelled before publication"));
        }
        let ctx = format!("\"_rustic_ctx\":{}", serde_json::to_string(&id)?);
        return Ok(match out.json.strip_prefix('{') {
            Some(rest) if rest.starts_with('}') => format!("{{{ctx}{rest}"),
            Some(rest) => format!("{{{ctx},{rest}"),
            None => out.json,
        });
    }
    // A re-run for a shard without context counts as its fetch phase (its
    // query phase counted the query and its usage).
    req.refetch = qtf == Some(crate::InternalQtf::Rerun);
    // A coordinator's single-round search of a small page: the coordinator
    // counts the fetch on the shards whose hits make its page.
    let single_round = qtf == Some(crate::InternalQtf::QueryFetch);
    req.defer_fetch_stats = single_round;
    // Shard request cache (`size: 0` requests). A single round reads no
    // cache, as the query phase it replaces does not.
    let cache_key = if single_round || req.profile || req.dfs_stats.is_some() || req.admitted_dfs_stats.is_some() || targets.iter().any(|t| t.is_security_restricted()) { None } else { crate::request_cache::key(&targets, orig_body, params, req.size) };
    if let Some((k, _)) = &cache_key {
        if let Some(hit) = crate::request_cache::get(k, &targets, start.elapsed().as_millis()) {
            return Ok(hit);
        }
    }
    let before = if req.profile { crate::obs_search::profile_snapshot(&targets) } else { Vec::new() };
    let out = search::execute(&targets, &req, &ExecOpts { terms_lookup: Some(&lookup), cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None })?;
    if !req.refetch {
        crate::usage::record(&targets, body);
    }
    if let Some((k, lookups)) = cache_key {
        crate::request_cache::put(k, lookups, &out.json);
    }
    if req.profile {
        return crate::obs_search::inject_profile(out.json, body, params, &targets, &before, index, &req.raw_search_scripts);
    }
    if single_round {
        return Ok(with_leading_member(out.json, &format!("\"{}\":{}", crate::QTF_FETCH_NS_KEY, out.fetch_nanos)));
    }
    Ok(out.json)
}

/// `json` (an object) with `member` (`"key":value`) inserted first.
fn with_leading_member(json: String, member: &str) -> String {
    match json.strip_prefix('{') {
        Some(rest) if rest.starts_with('}') => format!("{{{member}{rest}"),
        Some(rest) => format!("{{{member},{rest}"),
        None => json,
    }
}

/// Who a query_then_fetch reader context belongs to: the authenticated
/// subject (as ES's `SecurityContext` compares it: authentication type,
/// user and realm or API key id, and a run-as target), the request's index
/// expression and its cluster shard.
fn qtf_principal(user: &elasticsearch_engine::security::Authn) -> String {
    let id = |u: &elasticsearch_engine::security::Authn| {
        format!("{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}", u.auth_type, u.username, u.realm_name, u.realm_type, u.api_key_id().unwrap_or(""))
    };
    match &user.run_as {
        Some(target) => format!("{}\u{1e}{}", id(user), id(target)),
        None => id(user),
    }
}

fn fetch_owner(access: Option<&ReadAccess>, index: Option<&str>) -> elasticsearch_engine::node::FetchOwner {
    elasticsearch_engine::node::FetchOwner { principal: access.map(|a| qtf_principal(a.user())), index: index.map(str::to_string), shard: search::selected_read_shard() }
}

/// `{"ctx": id, "docs": [[target, leaf, doc, score], ...]}` of a
/// query_then_fetch fetch phase.
fn qtf_fetch_spec(f: &Value) -> Option<(String, Vec<(usize, usize, u32, f32)>)> {
    let ctx = f.get("ctx")?.as_str()?.to_string();
    let docs = f.get("docs")?.as_array()?.iter().map(|d| {
        let d = d.as_array()?;
        Some((d.first()?.as_u64()? as usize, d.get(1)?.as_u64()? as usize, u32::try_from(d.get(2)?.as_u64()?).ok()?, d.get(3).and_then(Value::as_f64).unwrap_or(0.0) as f32))
    }).collect::<Option<Vec<_>>>()?;
    Some((ctx, docs))
}

/// The child searches of a coordinator's rank plan phase: cluster
/// `_search`es over the request's targets.
struct ClusterChildSearch {
    rest: crate::ClusterRest,
    /// Several children at once, when the coordinator offers it.
    many: Option<crate::ClusterRestMany>,
    uri: String,
}

impl elasticsearch_engine::retriever::ChildSearch for ClusterChildSearch {
    fn search(&self, body: Value) -> Result<Value> {
        // The plan phase's task was cancelled (the client went away, or
        // `_tasks/_cancel`): no further child search starts. A running
        // child is the client task's child and stops with it.
        elasticsearch_engine::search::check_cancelled()?;
        (self.rest.0)("POST", &self.uri, Some(body))
    }

    fn search_all(&self, bodies: Vec<Value>) -> Vec<Result<Value>> {
        let Some(many) = self.many.as_ref().filter(|_| bodies.len() > 1) else {
            return bodies.into_iter().map(|b| self.search(b)).collect();
        };
        if let Err(e) = elasticsearch_engine::search::check_cancelled() {
            return bodies.iter().map(|_| Err(e.clone())).collect();
        }
        (many.0)(bodies.into_iter().map(|b| ("POST".to_string(), self.uri.clone(), Some(b))).collect())
    }
}

/// `/{index}/_search?<target selection params>` of a child search.
fn child_search_uri(index: Option<&str>, params: &HashMap<String, String>) -> String {
    fn enc(s: &str) -> String {
        let mut out = String::new();
        for b in s.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'*' | b',' | b':' => out.push(b as char),
                _ => out.push_str(&format!("%{b:02X}")),
            }
        }
        out
    }
    let mut q: Vec<String> = Vec::new();
    for k in ["routing", "preference", "ignore_unavailable", "allow_no_indices", "expand_wildcards", "ignore_throttled"] {
        if let Some(v) = params.get(k) {
            q.push(format!("{k}={}", enc(v)));
        }
    }
    let path = match index {
        Some(i) if !i.is_empty() => format!("/{}/_search", enc(i)),
        _ => "/_search".to_string(),
    };
    if q.is_empty() { path } else { format!("{path}?{}", q.join("&")) }
}

thread_local! {
    /// The hit spans ([`crate::encode_hit_spans`]) of the query_then_fetch
    /// fetch phase [`run_search_inner`] just rendered on this thread.
    static FETCH_HIT_SPANS: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

fn search(node: &Node, req: &Req, index: Option<&str>) -> R {
    let started = std::time::Instant::now();
    FETCH_HIT_SPANS.with(|s| s.borrow_mut().take());
    let (body,raw)=match req.json_shared() {
        Ok(body)=>(body,None),
        Err(error)=>match std::str::from_utf8(req.body_text()).ok().and_then(|s|elasticsearch_engine::source_json::search_request(s).ok()) {
            Some((parsed,raw))=>(Arc::new(parsed),Some(raw)), None=>return Err(error),
        }
    };
    // The request's own extensions unless a raw body joins them.
    let with_raw=raw.map(|raw|{let mut e=req.extensions.clone();e.insert(raw);e});
    let extensions=with_raw.as_ref().unwrap_or(&req.extensions);

    let _ = elasticsearch_engine::warnings::take();
    let ran = std::time::Instant::now();
    let json = run_search_ext(node, &body, &params_with(req), index, ResolveOpts::from_params(&req.params), req.access.as_deref(), Some(extensions))?;
    elasticsearch_engine::search::search_phases::record_named("engine_search_handler", vec![("parse", (ran - started).as_micros()), ("run", ran.elapsed().as_micros())]);
    let spans = FETCH_HIT_SPANS.with(|s| s.borrow_mut().take());
    let mut r = Resp::raw(200, json);
    // A query_then_fetch fetch phase (the coordinator asked its hit spans):
    // where its hits are, so the coordinator need not read them to find them.
    if let Some(spans) = spans.filter(|_| req.headers.contains_key(crate::HIT_SPANS_HEADER)) {
        r.headers.push((crate::HIT_SPANS_HEADER, spans));
    }
    for w in elasticsearch_engine::warnings::take() {
        r.headers.push(warning_header(&w));
    }
    Ok(r)
}

/// Deprecated `_knn_search` API: `{knn, filter, fields, _source, docvalue_fields,
/// stored_fields}` run as a `_search` with a top-level `knn` section (`size`
/// = `k`), with the deprecation warning ES emits.
fn knn_search(node: &Node, req: &Req, index: &str) -> R {
    let body = req.json()?;
    let Some(o) = body.as_object() else {
        return Err(EsError::parsing("[knn_search] request body must be an object"));
    };
    // `KnnSearchRequestParser` (ObjectParser `knn-search`).
    const ALLOWED: &[&str] = &["knn", "filter", "fields", "_source", "docvalue_fields", "stored_fields"];
    for (k, v) in o {
        if !ALLOWED.contains(&k.as_str()) {
            return Err(EsError::unknown_field_xc("knn-search", k, None, ALLOWED));
        }
        if k == "knn" && !v.is_object() {
            return Err(EsError::unsupported_type_xc("knn-search", "knn", v, None));
        }
    }
    let mut knn = match o.get("knn") {
        Some(Value::Object(k)) => k.clone(),
        _ => return Err(EsError::illegal_argument("missing required [knn] section in search body")),
    };
    {
        const KNN_FIELDS: &[&str] = &["field", "query_vector", "query_vector_builder", "k", "num_candidates", "similarity"];
        let wrap = |e: EsError| EsError::field_failure("knn-search", "knn", None, e);
        for (k, _) in &knn {
            if !KNN_FIELDS.contains(&k.as_str()) {
                return Err(wrap(EsError::unknown_field_xc("knn", k, Some("knn"), KNN_FIELDS)));
            }
        }
        let missing: Vec<&str> = ["k", "num_candidates"].into_iter().filter(|k| !knn.contains_key(*k)).collect();
        if !missing.is_empty() {
            return Err(wrap(EsError::illegal_argument(format!("Required [{}]", missing.join(", "))).at_key("knn", None).cursor("end")));
        }
    }
    let mut search = serde_json::Map::new();
    for (k, v) in o {
        match k.as_str() {
            "knn" => {}
            "filter" => {
                let filters: Vec<Value> = match v {
                    Value::Array(a) => a.clone(),
                    other => vec![other.clone()],
                };
                for f in &filters {
                    elasticsearch_engine::search_strict::check_query(f, "filter", None).map_err(|e| EsError::field_failure("knn-search", "filter", None, e))?;
                }
                if !filters.is_empty() {
                    knn.insert("filter".into(), Value::Array(filters));
                }
            }
            _ => {
                search.insert(k.clone(), v.clone());
            }
        }
    }
    if let Some(k) = knn.get("k").and_then(Value::as_u64) {
        search.insert("size".into(), Value::from(k));
    }
    search.insert("knn".into(), Value::Object(knn));
    let json = run_search_ext(node, &Value::Object(search), &params_with(req), Some(index), ResolveOpts::from_params(&req.params), req.access.as_deref(), Some(&req.extensions))?;
    let mut r = Resp::raw(200, json);
    r.headers.push((
        "Warning",
        format!("299 Elasticsearch-{}-{} \"The kNN search API has been replaced by the `knn` option in the search API.\"", elasticsearch_engine::node::ES_VERSION, crate::cluster::BUILD_HASH),
    ));
    Ok(r)
}

fn scroll(node: &Node, req: &Req) -> R {
    let body = if req.body_str().map(|s| s.trim_start().starts_with('{')).unwrap_or(true) { scroll_body(req, false)? } else { Value::Null };
    let id = body.get("scroll_id").and_then(Value::as_str).map(str::to_string).or_else(|| req.param("scroll_id").map(str::to_string)).or_else(|| {
        // Deprecated `/_search/scroll/{scroll_id}` path form.
        Some(req.seg(2)).filter(|s| !s.is_empty()).map(str::to_string)
    }).or_else(|| {
        // Raw scroll id as body.
        let s = req.body_str().ok()?.trim();
        (!s.is_empty() && !s.starts_with('{')).then(|| s.to_string())
    });
    let Some(id) = id else {
        return Err(EsError::action_validation(&["scrollId is missing"]));
    };
    let ka = body.get("scroll").and_then(Value::as_str).map(str::to_string).or_else(|| req.param("scroll").map(str::to_string));
    if let Some(k) = &ka {
        search_validate::check_keep_alive(node, keep_alive_named(Some(k), "scroll")?)?;
    }
    elasticsearch_engine::search_ids::parse_scroll_id(&id)?;
    let (targets, sreq, cursors, lost) = {
        let mut sc = node.scrolls.lock();
        // Expired is gone, whether or not the reaper has run yet.
        if sc.get(&id).is_some_and(|ctx| ctx.expires <= Instant::now()) {
            sc.remove(&id);
        }
        let ctx = sc.get_mut(&id).ok_or_else(|| {
            EsError::search_phase("", EsError::search_context_missing(&id)).with_status_404()
        })?;
        if let Some(k) = &ka {
            ctx.keep_alive = keep_alive(Some(k))?;
        }
        ctx.expires = Instant::now() + ctx.keep_alive;
        if req.extensions.get::<crate::InternalScrollTouch>().is_some() && !crate::SCROLL_TOUCH_AS_PAGE.load(std::sync::atomic::Ordering::Relaxed) {
            // A composite scroll's page that needs nothing from this
            // sub-context still keeps it alive.
            return Ok(Resp::ok(json!({"_scroll_id": id, "renewed": true})));
        }
        (ctx.targets.clone(), ctx.req.clone(), ctx.cursors.clone(), ctx.lost.clone())
    };
    let lookup = terms_lookup(node, req.access.as_deref());
    let mut sreq = sreq;
    if req.param("rest_total_hits_as_int") == Some("true") {
        sreq.rest_total_hits_as_int = true;
    }
    let out = search::execute(&targets, &sreq, &ExecOpts { terms_lookup: Some(&lookup), cursors: Some(&cursors), max_buckets: node.max_buckets(), scroll_id: Some(id.clone()), pit_id: None })?;
    if let Some(ctx) = node.scrolls.lock().get_mut(&id) {
        ctx.cursors = out.cursors;
    }
    // A composite scroll's sub page (`HIT_SPANS_HEADER` asked): where its
    // hits are, so the coordinator cuts them out without reading the page.
    let spans = (lost.is_empty() && req.headers.contains_key(crate::HIT_SPANS_HEADER)).then_some(out.hit_spans.as_ref()).flatten()
        .map(|(end, lens)| crate::encode_hit_spans(out.json.len() - end, lens));
    let mut r = Resp::raw(200, elasticsearch_engine::node::report_lost_targets(out.json, &lost));
    if let Some(spans) = spans {
        r.headers.push((crate::HIT_SPANS_HEADER, spans));
    }
    Ok(r)
}

trait Status404 {
    fn with_status_404(self) -> Self;
}
impl Status404 for EsError {
    fn with_status_404(mut self) -> Self {
        self.status = 404;
        self
    }
}

fn clear_scroll(node: &Node, req: &Req) -> R {
    let all = req.seg(2) == "_all";
    let mut ids: Vec<String> = Vec::new();
    if !all {
        if let Some(s) = req.seg(2).get(0..).filter(|s| !s.is_empty()) {
            ids.extend(s.split(',').map(str::to_string));
        }
        if let Some(p) = req.param("scroll_id") {
            ids.extend(p.split(',').map(str::to_string));
        }
        let body = if req.body.is_empty() { Value::Null } else { scroll_body(req, true)? };
        // `ClearScrollRequest.fromXContent` resets the ids: a body overrides the path / query string.
        if body.is_object() {
            ids.clear();
        }
        match body.get("scroll_id") {
            Some(Value::String(s)) => ids.push(s.clone()),
            Some(Value::Array(a)) => ids.extend(a.iter().map(elasticsearch_engine::query::dsl::scalar_string)),
            Some(other) => ids.push(elasticsearch_engine::query::dsl::scalar_string(other)),
            None => {}
        }
        if ids.is_empty() {
            return Err(EsError::action_validation(&["no scroll ids specified"]));
        }
        for id in &ids {
            elasticsearch_engine::search_ids::parse_scroll_id(id)?;
        }
    }
    let mut sc = node.scrolls.lock();
    let freed = if all {
        let n = sc.len();
        sc.clear();
        n
    } else {
        ids.iter().filter(|i| sc.remove(*i).is_some()).count()
    };
    let status = if !all && freed == 0 { 404 } else { 200 };
    Ok(Resp::json(status, json!({"succeeded": true, "num_freed": freed})))
}

fn open_pit(node: &Node, req: &Req, index: &str) -> R {
    let ka = req.param("keep_alive").ok_or_else(|| EsError::action_validation(&["[keep_alive] is not specified"]))?;
    let ka = keep_alive(Some(ka))?;
    // `search.max_keep_alive` bounds a point in time as it bounds a scroll.
    search_validate::check_keep_alive(node, ka)?;
    // `OpenPointInTimeRequest` body (ObjectParser): only `index_filter`.
    let body = req.json()?;
    if let Some(o) = body.as_object() {
        for (k, v) in o {
            if k != "index_filter" {
                return Err(EsError::unknown_field_xc("open_point_in_time_request", k, None, &["index_filter"]));
            }
            elasticsearch_engine::search_strict::check_query(v, "index_filter", None).map_err(|e| EsError::field_failure("open_point_in_time_request", "index_filter", None, e))?;
        }
    }
    // Opening a PIT accepts cross-cluster expressions: plan the remote parts.
    let targets = restrict(req.access.as_deref(), elasticsearch_engine::ccr::remote::resolve_local_part(node, index, ResolveOpts::from_params(&req.params))?.0)?;
    let targets = apply_search_routing(targets, req.param("routing"));
    if whole_local_read(Some(&req.extensions)) {
        search::require_local_shards(&targets)?;
    }
    let mut snap = node.snapshot(&targets)?;
    // `index_filter`: keep only the indices that have a matching document.
    if let Some(f) = body.get("index_filter") {
        let lookup = terms_lookup(node, req.access.as_deref());
        snap.retain(|t| search::count(std::slice::from_ref(t), Some(f), Some(&lookup)).map(|c| c > 0).unwrap_or(true));
    }
    let n = snap.len();
    let id = new_pit_id(&snap);
    let accounting = Some(elasticsearch_engine::node::context_charge(&snap, 0));
    node.pits.lock().insert(id.clone(), PitCtx { targets: snap, keep_alive: ka, expires: Instant::now() + ka, accounting, lost: Vec::new() });
    Ok(Resp::ok(json!({"id": id, "_shards": {"total": n, "successful": n, "skipped": 0, "failed": 0}})))
}

fn close_pit(node: &Node, req: &Req) -> R {
    if req.body.is_empty() && req.param("source").is_none() {
        return Err(EsError::new(400, "parse_exception", "request body or source parameter is required"));
    }
    let body = req.json()?;
    // `ClosePointInTimeRequest.fromXContent`: `{"id": "<string>"}` only.
    let mut id: Option<String> = None;
    if let Some(o) = body.as_object() {
        for (k, v) in o {
            if k != "id" {
                return Err(EsError::illegal_argument(format!("Unknown parameter [{k}] in request body or parameter is of the wrong type[FIELD_NAME] ")));
            }
            match v {
                Value::Object(_) | Value::Array(_) => return Err(EsError::illegal_argument("the request must contain only [id field")),
                other => id = Some(elasticsearch_engine::query::dsl::scalar_string(other)),
            }
        }
    }
    let id = id.ok_or_else(|| EsError::illegal_argument("search context id is is not provided"))?;
    elasticsearch_engine::search_ids::parse_pit_id(&id)?;
    let id = id.as_str();
    let found = node.pits.lock().remove(id).is_some();
    if !found {
        return Ok(Resp::json(404, json!({"succeeded": true, "num_freed": 0})));
    }
    Ok(Resp::ok(json!({"succeeded": true, "num_freed": 1})))
}

/// `RestActions.getQueryContent`: a body syntax error is reported as
/// `parsing_exception "Failed to parse"` at the parser's current token.
pub(crate) fn query_body(req: &Req) -> Result<Value> {
    let text = req.body_text();
    if text.iter().all(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r')) {
        return Ok(Value::Null);
    }
    if let Some(v) = elasticsearch_engine::jsonpos::parse_plain(text).filter(|v| !v.is_null()) {
        return Ok(v);
    }
    if let Err(e) = elasticsearch_engine::jsonpos::validate(text) {
        return Err(e.search_failure(text));
    }
    req.json()
}

fn count(node: &Node, req: &Req, index: Option<&str>) -> R {
    let (body,raw)=match query_body(req) {
        Ok(body)=>(body,None),
        Err(error)=>match std::str::from_utf8(req.body_text()).ok().and_then(|s|elasticsearch_engine::source_json::search_request(s).ok()) {
            Some(parsed)=>parsed,None=>return Err(error),
        },
    };
    let scripts=raw.as_ref().map(elasticsearch_engine::source_json::search_scripts).unwrap_or_default();
    if let Some(o) = body.as_object() {
        for (k, v) in o {
            if k != "query" {
                return Err(EsError::parsing(format!("request does not support [{k}]")).at_key(k, None).cursor("key").with_lc());
            }
            let mut environment=elasticsearch_engine::script::ScriptEnv::new(None,None);
            environment.raw_search_scripts=scripts.clone();
            elasticsearch_engine::script::with_env(environment,||elasticsearch_engine::search_strict::check_query(v,"query",None))?;
        }
    }
    let mut terminate_after = false;
    if let Some(t) = req.param("terminate_after") {
        let n = t.trim().parse::<i64>().map_err(|_| {
            EsError::illegal_argument(format!("Failed to parse int parameter [terminate_after] with value [{t}]")).caused_by(EsError::new(400, "number_format_exception", format!("For input string: \"{t}\"")))
        })?;
        if n < 0 {
            return Err(EsError::illegal_argument("terminateAfter must be > 0"));
        }
        terminate_after = n > 0;
    }
    let min_score = match req.param("min_score") {
        Some(v) => Some(v.parse::<f32>().map_err(|_| EsError::illegal_argument(format!("Failed to parse float parameter [min_score] with value [{v}]")))?),
        None => None,
    };
    let targets = restrict(req.access.as_deref(), targets_for(node, req, index)?)?;
    let targets = apply_search_routing(targets, req.param("routing"));
    if whole_local_read(Some(&req.extensions)) {
        search::require_local_shards(&targets)?;
    }
    let targets = search::expand_targets(&targets)?;
    let mut query = body.get("query").cloned();
    if let Some(q) = req.param("q") {
        let mut qs = serde_json::Map::new();
        qs.insert("query".into(), json!(q));
        for (p, k) in [("df", "default_field"), ("default_operator", "default_operator"), ("analyzer", "analyzer")] {
            if let Some(v) = req.param(p) {
                qs.insert(k.into(), json!(v));
            }
        }
        for (p, k) in [("lenient", "lenient"), ("analyze_wildcard", "analyze_wildcard")] {
            if let Some(v) = req.param(p) {
                qs.insert(k.into(), json!(v == "true" || v.is_empty()));
            }
        }
        query = Some(json!({ "query_string": qs }));
    }
    let lookup = terms_lookup(node, req.access.as_deref());
    let n = targets.len();
    let mut sreq = SearchRequest { query, size: 0, track_total_hits: Some(elasticsearch_engine::rustic_core::query::TrackTotalHits::True), ..Default::default() };
    sreq.min_score = min_score;
    sreq.raw_search_scripts=scripts;
    let out = search::execute(&targets, &sreq, &ExecOpts { terms_lookup: Some(&lookup), cursors: None, max_buckets: node.max_buckets(), scroll_id: None, pit_id: None })?;
    // ES's count never terminates early; it only reports `terminated_early: false` when asked to.
    let mut resp = serde_json::Map::new();
    if terminate_after {
        resp.insert("terminated_early".into(), json!(false));
    }
    resp.insert("count".into(), json!(out.total));
    resp.insert("_shards".into(), json!({"total": n, "successful": n, "skipped": 0, "failed": 0}));
    Ok(Resp::ok(Value::Object(resp)))
}

fn msearch(node: &Node, req: &Req, index: Option<&str>) -> R {
    let start = Instant::now();
    let body = req.body_str()?;
    let lines: Vec<&str> = body.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut responses: Vec<String> = Vec::new();
    // `TransportMultiSearchAction`: buffered sub-responses are accounted on
    // the request breaker; a trip fails the whole request with one 429.
    let mut breaker = MsearchBreaker::default();
    let mut i = 0;
    let cancelled = req.extensions.get::<crate::ParentCancelled>().map(|c| c.0.clone());
    while i < lines.len() {
        // A cancelled coordinator stops waiting; stop at the item boundary.
        if cancelled.as_ref().is_some_and(|c| c()) {
            return Err(EsError::new(400, "task_cancelled_exception", "The parent task was cancelled, shouldn't start any child tasks, by user request"));
        }
        let header: Value = crate::parse_body(lines[i].as_bytes())?;
        if let Some(h) = header.as_object() {
            for k in h.keys() {
                if !matches!(k.as_str(), "index" | "indices" | "search_type" | "request_cache" | "preference" | "routing" | "allow_partial_search_results" | "ccs_minimize_roundtrips" | "ignore_unavailable" | "allow_no_indices" | "expand_wildcards" | "ignore_throttled" | "cancel_after_time_interval") {
                    return Err(EsError::illegal_argument(format!("key [{k}] is not supported in the metadata section")));
                }
            }
        }
        let line_text = lines.get(i + 1).copied();
        let (body, raw) = match line_text {
            Some(l) => match crate::parse_body(l.as_bytes()) {
                Ok(body)=>(body,None),
                Err(error)=>elasticsearch_engine::source_json::search_request(l).map_err(|_|error)?,
            },
            None => return Err(EsError::illegal_argument("msearch request must be terminated by a newline [\n]")),
        };
        // Positions in a sub-request's errors are relative to its own line.
        let line_text = line_text.unwrap_or("");
        i += 2;
        let idx: Option<String> = match header.get("index") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Array(a)) => Some(a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(",")),
            _ => index.map(str::to_string),
        };
        if body.get("pit").is_some() && header.get("index").is_some() {
            return Err(EsError::action_validation(&["[indices] cannot be used with point in time. Do not specify any index with point in time."]));
        }
        let mut params: HashMap<String, String> = HashMap::new();
        for k in ["rest_total_hits_as_int", "typed_keys"] {
            if let Some(v) = req.param(k) {
                params.insert(k.into(), v.into());
            }
        }
        let mut hp = params.clone();
        if let Some(o) = header.as_object() {
            for (k, v) in o {
                if matches!(k.as_str(), "ignore_unavailable" | "allow_no_indices" | "expand_wildcards") {
                    hp.insert(k.clone(), elasticsearch_engine::query::dsl::scalar_string(v));
                }
            }
        }
        let resolve = ResolveOpts::from_params(&hp);
        // ES parses every sub-request body up front: a structural error fails the whole request.
        let mut sub_params = params.clone();
        sub_params.remove("rest_total_hits_as_int");
        let mut extensions=req.extensions.clone();
        let mut environment=elasticsearch_engine::script::ScriptEnv::new(None,None);
        if let Some(raw)=raw {
            environment.raw_search_scripts=elasticsearch_engine::source_json::search_scripts(&raw);
            extensions.insert(raw);
        }
        elasticsearch_engine::script::with_env(environment,||search::parse_request(&body,&sub_params))
            .map_err(|e| elasticsearch_engine::jsonpos::resolve(e,line_text.as_bytes()))?;
        match run_search_ext(node, &body, &params, idx.as_deref(), resolve, req.access.as_deref(), Some(&extensions)) {
            Ok(mut s) => {
                breaker.add(node.breakers, &s, responses.iter().map(String::as_str))?;
                s.pop();
                s.push_str(",\"status\":200}");
                responses.push(s);
            }
            // ES validates `rest_total_hits_as_int` while parsing the whole request.
            Err(e) if e.reason.starts_with("[rest_total_hits_as_int] cannot be used") => return Err(e),
            Err(e) => {
                let e = elasticsearch_engine::jsonpos::resolve(e, line_text.as_bytes());
                responses.push(json!({"error": e.error_object(false), "status": e.status}).to_string())
            }
        }
    }
    Ok(Resp::raw(200, format!("{{\"took\":{},\"responses\":[{}]}}", start.elapsed().as_millis(), responses.join(","))))
}

/// Request-breaker accounting of an msearch's buffered sub-responses
/// (released when the msearch response is built).
#[derive(Default)]
pub struct MsearchBreaker {
    /// Exact `estimateActualBytes` total held on the request breaker, once
    /// the cheap bound got close to the breaker's headroom (`None` before).
    held: Option<elasticsearch_engine::memory::Reservation>,
    /// Upper bound of the estimate from the response sizes alone, so the
    /// common case never parses a response: every hit is >= 40 bytes of
    /// JSON and is estimated at 400 bytes plus its source (and fields).
    bound: u64,
}

impl MsearchBreaker {
    pub fn add<'a>(&mut self, b: &'static elasticsearch_engine::memory::Breakers, resp: &str, previous: impl Iterator<Item = &'a str>) -> Result<()> {
        const LABEL: &str = "<msearch_response>";
        if self.held.is_none() {
            self.bound = self.bound.saturating_add(512).saturating_add((resp.len() as u64).saturating_mul(11));
            let Ok(limit) = u64::try_from(b.request.limit()) else { return Ok(()) };
            let headroom = (limit as f64 / b.request.overhead().max(f64::MIN_POSITIVE)) as u64;
            if self.bound <= headroom.saturating_sub(b.request.used()) {
                return Ok(());
            }
            let before: u64 = previous.map(msearch_response_bytes).fold(0u64, u64::saturating_add);
            self.held = Some(b.reserve(elasticsearch_engine::memory::Kind::Request, before, LABEL)?);
        }
        let bytes = msearch_response_bytes(resp);
        if let Some(h) = self.held.as_mut() {
            h.grow(bytes, LABEL)?;
        }
        Ok(())
    }
}

/// `TransportMultiSearchAction.estimateActualBytes` of a rendered response.
fn msearch_response_bytes(resp: &str) -> u64 {
    let Ok(v) = serde_json::from_str::<Value>(resp) else { return 512 };
    fn value_bytes(v: &Value) -> u64 {
        match v {
            Value::String(s) => 32 + s.len() as u64,
            Value::Number(_) => 24,
            _ => 32,
        }
    }
    fn hit_bytes(h: &Value) -> u64 {
        let mut b = 400;
        if let Some(src) = h.get("_source") {
            b += serde_json::to_string(src).map(|s| s.len() as u64).unwrap_or(0);
        }
        if let Some(Value::Object(f)) = h.get("fields") {
            b += f.len() as u64 * 200;
            b += f.values().flat_map(|x| x.as_array().cloned().unwrap_or_default()).map(|x| value_bytes(&x)).sum::<u64>();
        }
        if let Some(Value::Array(sort)) = h.get("sort") {
            b += 64 + sort.len() as u64 * 8 + 2 * sort.iter().map(value_bytes).sum::<u64>();
        }
        if let Some(Value::Object(ih)) = h.get("inner_hits") {
            for x in ih.values() {
                for ih in x.pointer("/hits/hits").and_then(Value::as_array).into_iter().flatten() {
                    b += hit_bytes(ih);
                }
            }
        }
        b
    }
    let mut bytes = 512;
    for h in v.pointer("/hits/hits").and_then(Value::as_array).into_iter().flatten() {
        bytes += hit_bytes(h);
    }
    bytes
}

/// The enterprise-licensed rank features (`RRFRankPlugin.RANK_RRF_FEATURE`,
/// `LINEAR_RETRIEVER_FEATURE`): the legacy `rank.rrf` and the `rrf` /
/// `linear` retrievers anywhere in the retriever tree.
pub fn rank_license(node: &Node, body: &Value) -> Result<()> {
    use elasticsearch_engine::license::{check, Mode};
    fn walk(r: &Value, out: &mut Vec<&'static str>) {
        let Some(o) = r.as_object() else { return };
        for (kind, b) in o {
            match kind.as_str() {
                "rrf" => out.push("Reciprocal Rank Fusion (RRF)"),
                "linear" => out.push("linear retriever"),
                _ => {}
            }
            for x in b.get("retrievers").and_then(Value::as_array).into_iter().flatten() {
                walk(x.get("retriever").filter(|_| x.get("retriever").is_some_and(Value::is_object) && x.as_object().is_some_and(|m| m.len() <= 2)).unwrap_or(x), out);
            }
            if let Some(c) = b.get("retriever") {
                walk(c, out);
            }
        }
    }
    let mut features = Vec::new();
    if body.get("rank").and_then(|r| r.get("rrf")).is_some() {
        features.push("Reciprocal Rank Fusion (RRF)");
    }
    if let Some(r) = body.get("retriever") {
        walk(r, &mut features);
    }
    for f in features {
        check(node, f, Mode::Enterprise)?;
    }
    Ok(())
}

#[cfg(test)]
mod coordinated_search_tests {
    use super::*;

    /// A query_then_fetch reader context is fetched and freed only by its
    /// opener (user, realm, API key, run-as), for its index and shard.
    #[test]
    fn qtf_contexts_belong_to_their_requester_index_and_shard() {
        use elasticsearch_engine::security::{Authn, Principal, Security};
        let alice = Authn { username: "alice".into(), realm_name: "native1".into(), realm_type: "native".into(), auth_type: "realm", principal: Principal::User, run_as: None, ..Security::system() };
        let bob = Authn { username: "bob".into(), ..alice.clone() };
        let other_realm = Authn { realm_name: "file1".into(), realm_type: "file".into(), ..alice.clone() };
        let key = |id: &str| Authn { auth_type: "api_key", principal: Principal::ApiKey { id: id.into(), name: "k".into(), metadata: json!({}) }, ..alice.clone() };
        let as_bob = Authn { run_as: Some(Box::new(bob.clone())), ..alice.clone() };
        let ids = [&alice, &bob, &other_realm, &key("k1"), &key("k2"), &as_bob].map(qtf_principal);
        for (i, a) in ids.iter().enumerate() {
            for (j, b) in ids.iter().enumerate() {
                assert_eq!(a == b, i == j, "{a:?} vs {b:?}");
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("t", &json!({"mappings": {"properties": {"body": {"type": "text"}}}})).unwrap();
        let w = (0..5).map(|i| elasticsearch_engine::index::WriteReq::Index { id: Some(i.to_string()), source: json!({"body": "apple"}).to_string(), routing: None, cond: Default::default() }).collect();
        for r in node.get_index("t").unwrap().write(w, true) {
            r.unwrap();
        }
        node.get_index("t").unwrap().refresh().unwrap();
        let mut query = axum::http::Extensions::new();
        query.insert(crate::InternalQtf::Query);
        let mut fetch = axum::http::Extensions::new();
        fetch.insert(crate::InternalQtf::Fetch);
        let params = HashMap::new();
        let body = json!({"query": {"match": {"body": "apple"}}, "size": 3, "_source": false});
        let run = |body: &Value, index: &str, ext: &axum::http::Extensions| run_search_inner(&node, body, &params, Some(index), ResolveOpts::default(), None, Some(ext), None);
        let open = |shard: Option<u32>| -> (String, Vec<Value>) {
            let _scope = elasticsearch_engine::search::ReadShardScope::enter(shard);
            let v: Value = serde_json::from_str(&run(&body, "t", &query).unwrap()).unwrap();
            let docs = v["hits"]["hits"].as_array().unwrap().iter().map(|h| {
                let mut d = h["_rustic_doc"].as_array().unwrap().clone();
                d.push(h["_score"].clone());
                Value::Array(d)
            }).collect();
            (v["_rustic_ctx"].as_str().unwrap().to_string(), docs)
        };
        let fetch_body = |ctx: &str, docs: &[Value]| json!({"query": {"match": {"body": "apple"}}, "_rustic_fetch": {"ctx": ctx, "docs": docs}});
        let try_fetch = |ctx: &str, docs: &[Value], index: &str, shard: Option<u32>| {
            let _scope = elasticsearch_engine::search::ReadShardScope::enter(shard);
            run(&fetch_body(ctx, docs), index, &fetch)
        };
        let (ctx, docs) = open(Some(0));
        assert_eq!(docs.len(), 3);
        // Another shard or index: missing, and the context survives.
        for (index, shard) in [("t", Some(1)), ("t", None), ("t*", Some(0))] {
            let e = try_fetch(&ctx, &docs, index, shard).unwrap_err();
            assert_eq!(e.kind, "search_context_missing_exception", "{index} {shard:?}");
            assert!(try_fetch(&ctx, &[], index, shard).is_err(), "free from {index} {shard:?}");
        }
        // Another principal neither fetches nor frees it.
        let owner = elasticsearch_engine::node::FetchOwner { principal: Some(ids[0].clone()), index: Some("t".into()), shard: Some(0) };
        node.fetch_contexts.lock().get_mut(&ctx).unwrap().owner = owner.clone();
        assert!(try_fetch(&ctx, &docs, "t", Some(0)).is_err());
        assert!(node.take_fetch_context(&ctx, &elasticsearch_engine::node::FetchOwner { principal: Some(ids[1].clone()), ..owner.clone() }).is_none());
        assert!(node.fetch_contexts_bytes() > 0);
        // Its estimate is on the accounting breaker too, once.
        assert!(node.fetch_contexts.lock().values().all(|c| c.accounting.bytes() == c.bytes));
        assert!(node.take_fetch_context(&ctx, &owner).is_some());
        assert!(node.take_fetch_context(&ctx, &owner).is_none());
        // The opener fetches its docs once; the fetch frees the context.
        let (ctx, docs) = open(Some(0));
        let v: Value = serde_json::from_str(&try_fetch(&ctx, &docs, "t", Some(0)).unwrap()).unwrap();
        assert_eq!(v["hits"]["hits"].as_array().unwrap().len(), 3);
        assert!(try_fetch(&ctx, &docs, "t", Some(0)).is_err());
        assert_eq!(node.fetch_contexts_bytes(), 0);
        // An explicit free.
        let (ctx, _) = open(Some(0));
        assert_eq!(try_fetch(&ctx, &[], "t", Some(0)).unwrap(), "{}");
        assert!(node.fetch_contexts.lock().is_empty());

        // Expired contexts are reaped; the live ones stay.
        open(Some(0));
        node.reap_fetch_contexts(std::time::Instant::now());
        assert_eq!(node.fetch_contexts.lock().len(), 1);
        node.reap_fetch_contexts(std::time::Instant::now() + elasticsearch_engine::node::FETCH_CTX_KEEP_ALIVE + std::time::Duration::from_secs(1));
        assert!(node.fetch_contexts.lock().is_empty());
        // `search.default_keep_alive` (dynamic) sets their keep-alive.
        assert_eq!(node.fetch_keep_alive(), elasticsearch_engine::node::FETCH_CTX_KEEP_ALIVE);
        node.cluster_settings.write().1.insert("search.default_keep_alive".into(), json!("2s"));
        assert_eq!(node.fetch_keep_alive(), std::time::Duration::from_secs(2));
        open(Some(0));
        node.reap_fetch_contexts(std::time::Instant::now() + std::time::Duration::from_secs(3));
        assert!(node.fetch_contexts.lock().is_empty());
        node.cluster_settings.write().1.remove("search.default_keep_alive");
        // Reopening (or closing, or deleting) the index drops its contexts.
        let (ctx, docs) = open(Some(0));
        node.reopen_index("t").unwrap();
        assert!(node.fetch_contexts.lock().is_empty());
        assert!(try_fetch(&ctx, &docs, "t", Some(0)).is_err());
        open(Some(0));
        node.drop_fetch_contexts("t");
        assert!(node.fetch_contexts.lock().is_empty());
        // Closing the index by any path (here IndexService::set_open, as the
        // cluster's metadata apply and ILM do): a fetch is refused and frees
        // the context; reaping drops the others.
        let svc = node.get_index("t").unwrap();
        let (ctx, docs) = open(Some(0));
        open(Some(0));
        svc.set_open(false).unwrap();
        assert_eq!(try_fetch(&ctx, &docs, "t", Some(0)).unwrap_err().kind, "search_context_missing_exception");
        assert_eq!(node.fetch_contexts.lock().len(), 1);
        node.reap_fetch_contexts(std::time::Instant::now());
        assert!(node.fetch_contexts.lock().is_empty());
        svc.set_open(true).unwrap();
        assert_eq!(node.fetch_budget.lock().held(), (0, 0));
        // A copy promoted since its query phase keeps its contexts, as in ES
        // (the copy stays open): they are read and survive reaping. One
        // removed, reset or reopened is not read; the cluster drops a copy's
        // contexts at once when it removes or resets it.
        let (ctx, docs) = open(Some(0));
        let sh = node.get_index("t").unwrap().shard(0).unwrap();
        sh.promote(sh.shard.primary_term() + 1).unwrap();
        let (ctx2, docs2) = open(Some(0));
        let (ctx3, _) = open(Some(0));
        node.reap_fetch_contexts(std::time::Instant::now());
        assert_eq!(node.fetch_contexts.lock().len(), 3);
        for (c, d) in [(&ctx, &docs), (&ctx2, &docs2)] {
            let v: Value = serde_json::from_str(&try_fetch(c, d, "t", Some(0)).unwrap()).unwrap();
            assert_eq!(v["hits"]["hits"].as_array().unwrap().len(), 3);
        }
        node.drop_fetch_contexts_of_copy("t", 1);
        assert!(node.fetch_contexts.lock().contains_key(&ctx3));
        node.drop_fetch_contexts_of_copy("t", 0);
        assert!(node.fetch_contexts.lock().is_empty());
        // A shard without hits keeps no context (nothing to fetch).
        {
            let _scope = elasticsearch_engine::search::ReadShardScope::enter(Some(0));
            let v: Value = serde_json::from_str(&run(&json!({"query": {"match": {"body": "pear"}}, "size": 3, "_source": false}), "t", &query).unwrap()).unwrap();
            assert!(v.get("_rustic_ctx").is_none(), "{v}");
        }
        assert!(node.fetch_contexts.lock().is_empty());
        // A free ahead of its query phase (a copy the coordinator gave up
        // on): the late phase registers nothing; the id the coordinator
        // chose is used.
        let any = elasticsearch_engine::node::FetchOwner { principal: None, index: None, shard: None };
        assert!(try_fetch("late", &[], "t", Some(0)).is_err());
        assert!(!node.register_fetch_context("late".into(), Vec::new(), any.clone()));
        assert!(node.fetch_freed.lock().is_empty());
        // An id in use is never replaced.
        assert!(node.register_fetch_context("dup".into(), Vec::new(), any.clone()));
        assert!(!node.register_fetch_context("dup".into(), Vec::new(), any.clone()));
        assert_eq!(node.fetch_contexts.lock().len(), 1);
        node.fetch_contexts.lock().clear();
        // A free racing a late registration by the same owner frees it.
        assert!(node.register_fetch_context("raced".into(), Vec::new(), any.clone()));
        node.free_unregistered_fetch_context("raced", &any);
        assert!(node.fetch_contexts.lock().is_empty() && node.fetch_freed.lock().is_empty());
        // A full tombstone table evicts its oldest.
        for i in 0..elasticsearch_engine::node::MAX_FETCH_CONTEXTS + 1 {
            node.free_unregistered_fetch_context(&format!("t{i}"), &any);
        }
        assert!(node.fetch_freed.lock().len() <= elasticsearch_engine::node::MAX_FETCH_CONTEXTS);
        assert!(node.fetch_freed.lock().contains_key(&format!("t{}", elasticsearch_engine::node::MAX_FETCH_CONTEXTS)));
        node.fetch_freed.lock().clear();
        {
            let _scope = elasticsearch_engine::search::ReadShardScope::enter(Some(0));
            let mut chosen = query.clone();
            chosen.insert(crate::InternalQtfCtx("mine".into()));
            let v: Value = serde_json::from_str(&run(&body, "t", &chosen).unwrap()).unwrap();
            assert_eq!(v["_rustic_ctx"], "mine", "{v}");
        }
        node.fetch_contexts.lock().clear();
        // No count cap (ES has none for query_then_fetch contexts): far more
        // than the old 4096 contexts are admitted while their bytes fit.
        for i in 0..elasticsearch_engine::node::MAX_FETCH_CONTEXTS + 100 {
            assert!(node.register_fetch_context(format!("filler{i}"), Vec::new(), any.clone()));
        }
        node.fetch_contexts.lock().clear();
        assert_eq!(node.fetch_budget.lock().held(), (0, 0));
        // A budget out of bytes refuses before executing the query. No
        // scoring can be duplicated by a coordinator fetch fallback.
        let default_budget = node.fetch_budget.lock().clone();
        *node.fetch_budget.lock() = Arc::new(elasticsearch_engine::node::FetchBudget::with_byte_limit(256 * 256));
        for i in 0..256 {
            assert!(node.register_fetch_context(format!("filler{i}"), Vec::new(), any.clone()));
        }
        assert!(!node.register_fetch_context("one more".into(), Vec::new(), any.clone()));
        let mut pinned_query = query.clone();
        pinned_query.insert(crate::InternalQtfPinned);
        {
            let _scope = elasticsearch_engine::search::ReadShardScope::enter(Some(0));
            let shard = node.get_index("t").unwrap().shard(0).unwrap();
            let before = shard.shard.counters.search_total.load(std::sync::atomic::Ordering::Relaxed);
            let error = run(&body, "t", &pinned_query).unwrap_err();
            assert_eq!(error.status, 429);
            assert_eq!(error.kind, "es_rejected_execution_exception");
            assert_eq!(shard.shard.counters.search_total.load(std::sync::atomic::Ordering::Relaxed), before,
                "capacity refusal must not execute the shard query");
            // An older coordinator (no pinned marker) gets the answer it
            // re-runs in one phase: the query's result, no context.
            let v: Value = serde_json::from_str(&run(&body, "t", &query).unwrap()).unwrap();
            assert!(v.get("_rustic_ctx").is_none(), "{v}");
            assert!(v["hits"]["hits"].as_array().is_some_and(|h| !h.is_empty()), "{v}");
        }
        assert_eq!(node.fetch_contexts.lock().len(), 256);
        assert_eq!(node.fetch_budget.lock().held().0, 256);
        // Dropped contexts give their charges back.
        node.fetch_contexts.lock().clear();
        assert_eq!(node.fetch_budget.lock().held(), (0, 0));
        *node.fetch_budget.lock() = default_budget;
        // Capacity is reserved without an observable, expiring context.
        // Abandoning a query before publication gives it back immediately.
        let admission = node.reserve_fetch_context(&[]).unwrap();
        assert!(node.fetch_contexts.lock().is_empty());
        assert_eq!(node.fetch_budget.lock().held().0, 1);
        drop(admission);
        assert_eq!(node.fetch_budget.lock().held(), (0, 0));
        // Once capacity becomes available, execute exactly one query and
        // fetch its retained snapshot without executing scoring again.
        let shard = node.get_index("t").unwrap().shard(0).unwrap();
        let before = shard.shard.counters.search_total.load(std::sync::atomic::Ordering::Relaxed);
        let (ctx, docs) = open(Some(0));
        assert_eq!(shard.shard.counters.search_total.load(std::sync::atomic::Ordering::Relaxed), before + 1);
        try_fetch(&ctx, &docs, "t", Some(0)).unwrap();
        assert_eq!(shard.shard.counters.search_total.load(std::sync::atomic::Ordering::Relaxed), before + 1);
    }

    #[test]
    fn coordinator_search_uri_encodes_params_and_keeps_expressions() {
        let params = HashMap::from([("ignore_unavailable".to_string(), "true".to_string()), ("q".to_string(), "a b&c".to_string())]);
        assert_eq!(coordinator_search_uri("logs-*,rollup", &params), "/logs-*,rollup/_search?ignore_unavailable=true&q=a%20b%26c");
        assert_eq!(coordinator_search_uri("i", &HashMap::new()), "/i/_search");
    }
}

#[cfg(test)]
mod msearch_response_breaker_tests {
    use super::*;
    #[test]
    fn coordinator_response_total_trips_and_releases_shared_estimate() {
        let breakers = elasticsearch_engine::memory::Breakers::leak_for_tests(2000);
        // This fixture measures accumulated response estimates, not the RSS of
        // the entire test process against its deliberately tiny private budget.
        breakers.apply_settings(&|key| (key == "indices.breaker.total.use_real_memory").then(|| "false".into()));
        let response = json!({"hits":{"hits":[{"_source":{"text":"x".repeat(100)}}]}}).to_string();
        let mut guard = MsearchBreaker::default();
        guard.add(breakers, &response, std::iter::empty()).unwrap();
        let admitted = msearch_response_bytes(&response);
        assert_eq!(breakers.request.used(), admitted);
        let error = guard.add(breakers, &response, std::iter::once(response.as_str())).unwrap_err();
        assert_eq!(error.status, 429);
        assert_eq!(error.kind, "circuit_breaking_exception");
        assert!(error.reason.contains("<msearch_response>"));
        assert_eq!(breakers.request.used(), admitted, "denied growth retains only the first response estimate");
        drop(guard);
        assert_eq!(breakers.request.used(), 0);
    }
}

#[cfg(test)]
mod msearch_cancellation_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[test]
    fn copy_local_batch_stops_at_the_next_item_once_its_parent_is_cancelled() {
        let dir = tempfile::tempdir().unwrap();
        let node = Arc::new(Node::open(dir.path()).unwrap());
        node.create_index("cancel-batch", &json!({})).unwrap();
        let checks = Arc::new(AtomicUsize::new(0));
        let seen = checks.clone();
        let mut extensions = axum::http::Extensions::new();
        // Cancelled after the first item started.
        extensions.insert(crate::ParentCancelled(Arc::new(move || seen.fetch_add(1, Ordering::SeqCst) >= 1)));
        let req = Req { method: Method::POST, path: "/cancel-batch/_msearch".into(), segs: vec!["cancel-batch".into(), "_msearch".into()],
            params: Default::default(), body: "{}\n{}\n{}\n{}\n{}\n{}\n".into(), headers: Default::default(),
            access: None, extensions, api_version: crate::RestApiVersion::V9 };
        let error = msearch(&node, &req, Some("cancel-batch")).err().expect("cancelled batch");
        assert_eq!((error.status, error.kind.as_str()), (400, "task_cancelled_exception"));
        assert_eq!(checks.load(Ordering::SeqCst), 2, "one item ran, the second was never started");
        let mut plain = req;
        plain.extensions = Default::default();
        assert_eq!(msearch(&node, &plain, Some("cancel-batch")).unwrap().status, 200);
    }
}
