//! `_fleet/_fleet_search` and `_fleet/_fleet_msearch` with
//! `wait_for_checkpoints` on a cluster (ES `TransportFleetSearchAction`):
//! the engine validates the request (target, checkpoint count,
//! `refresh_interval`) as it renders those errors; then the search runs
//! through the cluster with the checkpoints in [`WAIT_HEADER`], and each
//! copy a shard request reaches waits there until it has refreshed up to
//! its shard's checkpoint before it searches (ES `SearchService`'s wait in
//! the shard query). A copy that cannot (an unissued checkpoint, a timeout,
//! a stuck or unreachable copy) fails that request, so the shard goes to
//! its next copy as any shard failure does, and is reported failed only
//! when no copy answered (`allow_partial_search_results` as for a search).
use super::*;

/// Internal request header of a fleet search's shard requests:
/// `{"t": timeout_ms, "c": {index: [checkpoint per shard]}}`. Stripped from
/// client requests at the edge.
pub(super) const WAIT_HEADER: &str = "x-rustic-internal-fleet-wait";

/// On the copy a shard-level read (naming its copies) reaches: wait for
/// the fleet checkpoints of [`WAIT_HEADER`]; `Some(failure)` when this copy
/// cannot serve the shard (the coordinator tries the next copy).
pub(super) async fn wait_in_read(node: &ClusterNode, req: &RestReq) -> Option<RestResp> {
    let (_, v) = req.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(WAIT_HEADER))?;
    let spec: Value = serde_json::from_str(v).ok()?;
    let indices = copies::read_copies(&req.headers)?;
    let shard = crate::shard_engines::shard_of(&req.headers).unwrap_or(0);
    let timeout = Duration::from_millis(spec["t"].as_u64().unwrap_or(30_000));
    for index in indices {
        let Some(checkpoint) = spec["c"][&index][shard as usize].as_i64().filter(|c| *c >= 0) else { continue };
        let (Ok(copy), Some(svc)) = (copies::readable_copy(node, &index, shard), node.engine_for(shard).and_then(|e| e.get_index(&index))) else {
            return Some(copies::shard_not_found(node, &index, shard));
        };
        let (svc2, copy2) = (svc.clone(), copy.clone());
        let unissued = tokio::task::spawn_blocking(move || elasticsearch_engine::fleet::checkpoint_failure(&svc2, Some(&copy2), checkpoint)).await.ok().flatten();
        use elasticsearch_engine::error::EsError;
        if let Some(reason) = unissued {
            let e = EsError::search_phase(&index, EsError::illegal_argument(reason));
            return Some(RestResp::json(e.status, &e.to_json()));
        }
        if let Some(reason) = wait_caught_up(&svc, &copy, checkpoint, timeout).await {
            let e = EsError::search_phase(&index, EsError::new(429, "search_timeout_exception", reason));
            return Some(RestResp::json(e.status, &e.to_json()));
        }
    }
    None
}

/// A copy's wait for `checkpoint` (`fleet::caught_up`: refreshed and
/// globally acknowledged), polled without holding a thread, at most
/// `timeout` (capped at `fleet::MAX_WAIT`). The copy stays search-active so
/// its scheduled refresh runs; no refresh is forced, and an expired wait is
/// ES's timeout (`Some(reason)`). The wait ends with the request: a
/// coordinator that gives up stops it as its request is dropped, and the
/// deadline bounds it otherwise.
async fn wait_caught_up(svc: &Arc<elasticsearch_engine::index::IndexService>, copy: &Arc<elasticsearch_engine::index::IndexShard>, checkpoint: i64, timeout: Duration) -> Option<String> {
    use elasticsearch_engine::fleet;
    let timeout = timeout.min(fleet::MAX_WAIT);
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if fleet::caught_up(svc, copy, checkpoint) {
            return None;
        }
        if tokio::time::Instant::now() >= deadline {
            return Some(fleet::wait_timed_out(checkpoint, timeout));
        }
        copy.shard.note_search_wait();
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Rolling upgrades: a coordinator of the previous release asks each copy
/// to wait (`internal:fleet/wait_refreshed`) before it searches; this node
/// answers as that release did (`Some(reason)`: the shard's failure).
pub const ACTION_WAIT_REFRESHED: &str = "internal:fleet/wait_refreshed";

/// The previous release's wait request (see [`ACTION_WAIT_REFRESHED`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaitRefreshed {
    pub index: SharedText,
    pub shard: u32,
    pub checkpoint: i64,
    pub timeout_ms: u64,
}

pub(super) fn register(node: &Arc<ClusterNode>) {
    let n = node.clone();
    node.transport.register_handler(ACTION_WAIT_REFRESHED, move |r: WaitRefreshed| {
        let n = n.clone();
        async move {
            let not_here = || RemoteError::new("shard_not_found_exception", "no such shard", 404);
            let copy = copies::readable_copy(&n, &r.index, r.shard).map_err(|_| not_here())?;
            let svc = n.engine_for(r.shard).and_then(|e| e.get_index(&r.index)).ok_or_else(not_here)?;
            let (svc2, copy2) = (svc.clone(), copy.clone());
            let unissued = tokio::task::spawn_blocking(move || elasticsearch_engine::fleet::checkpoint_failure(&svc2, Some(&copy2), r.checkpoint))
                .await.map_err(|e| RemoteError::internal(e.to_string()))?;
            if unissued.is_some() {
                return Ok::<Option<String>, RemoteError>(unissued);
            }
            Ok(wait_caught_up(&svc, &copy, r.checkpoint, Duration::from_millis(r.timeout_ms)).await)
        }
    });
}

fn checkpoints_of(v: Option<&str>) -> Vec<i64> {
    v.map(|v| v.split(',').filter(|s| !s.trim().is_empty()).filter_map(|s| s.trim().parse().ok()).collect()).unwrap_or_default()
}

pub(super) async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> Option<RestResp> {
    let segs = req.segments();
    if !matches!(req.method(), Method::GET | Method::POST) {
        return None;
    }
    match segs.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        [index, "_fleet", "_fleet_search"] => {
            let params = req.params();
            let checkpoints = checkpoints_of(params.get("wait_for_checkpoints").map(String::as_str));
            if checkpoints.is_empty() || index.contains(',') {
                return None;
            }
            Some(fleet_search(node, req, index, &checkpoints).await)
        }
        [.., "_fleet", "_fleet_msearch"] => fleet_msearch(node, req, segs.first().filter(|_| segs.len() == 3).map(String::as_str)).await,
        _ => None,
    }
}

/// One fleet search with checkpoints (`req` is a `_fleet_search` request;
/// its validation errors are the engine's).
async fn fleet_search(node: &Arc<ClusterNode>, req: &RestReq, expr: &str, checkpoints: &[i64]) -> RestResp {
    let params = req.params();
    // The engine validates (and renders): the same request, nothing to wait
    // for, nothing to fetch.
    let none = vec!["-1"; checkpoints.len()].join(",");
    let mut probe = req.clone();
    let base = crate::rest::util::strip_params(&req.uri, &["wait_for_checkpoints"]);
    let sep = if base.contains('?') && !base.ends_with('?') { "&" } else if base.ends_with('?') { "" } else { "?" };
    if let Err(e)=probe.set_uri(&format!("{base}{sep}wait_for_checkpoints={none}")) {return RestResp::json(e.status,&e.to_json())}
    if let Err(e)=probe.set_body(br#"{"size":0}"#) {return RestResp::json(e.status,&e.to_json())}
    if let Err(e)=probe.set_method("POST") {return RestResp::json(e.status,&e.to_json())}
    if !probe.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-type")) {
        if let Err(e)=probe.push_header("content-type","application/json") {return RestResp::json(e.status,&e.to_json())}
    }
    let checked = local_canonical(node, &probe).await;
    if !checked.is_success() {
        return checked;
    }
    let engine = node.engine.clone();
    let target = expr.to_string();
    let index = match tokio::task::spawn_blocking(move || elasticsearch_engine::fleet::single_index(&engine, &target).map(|s| s.name.clone())).await {
        Ok(Ok(i)) => i,
        Ok(Err(e)) => return RestResp::json(e.status, &e.to_json()),
        Err(e) => return RestResp::error(500, "exception", &e.to_string()),
    };
    let timeout = params.get("wait_for_checkpoints_timeout").and_then(|t| crate::rest::parse_timeout(t)).unwrap_or(Duration::from_secs(30));
    // The search: every shard on its copies, each copy waiting for its
    // checkpoint before it searches.
    let uri = crate::rest::util::strip_params(&req.uri.replacen("/_fleet/_fleet_search", "/_search", 1), &["wait_for_checkpoints", "wait_for_checkpoints_timeout"]);
    let mut sub=req.clone();
    if let Err(e)=sub.set_uri(&uri) {return RestResp::json(e.status,&e.to_json())}
    let mut by_index = serde_json::Map::new();
    by_index.insert(index.clone(), json!(checkpoints));
    if let Err(e)=sub.replace_header(WAIT_HEADER,&json!({"t": timeout.as_millis() as u64, "c": by_index}).to_string()) {return RestResp::json(e.status,&e.to_json())}
    Box::pin(read(node, &sub, &index)).await
}

/// `_fleet_msearch` when an item waits for checkpoints: each item as a
/// fleet search of its own (`None`: no item waits; the engine serves it).
async fn fleet_msearch(node: &Arc<ClusterNode>, req: &RestReq, path_index: Option<&str>) -> Option<RestResp> {
    let started = std::time::Instant::now();
    let text = std::str::from_utf8(&req.body).ok()?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() % 2 != 0 {
        return None;
    }
    let mut items = Vec::new();
    let mut waits = false;
    for pair in lines.chunks(2) {
        let header: Value = serde_json::from_str(pair[0]).ok()?;
        let index = match header.get("index") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(","),
            _ => path_index.unwrap_or("_all").to_string(),
        };
        let cps: Vec<i64> = match header.get("wait_for_checkpoints") {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_i64).collect(),
            Some(Value::String(s)) => checkpoints_of(Some(s)),
            Some(Value::Number(n)) => n.as_i64().into_iter().collect(),
            _ => Vec::new(),
        };
        // A multi-index target that waits fails the whole request: the
        // engine renders it.
        if !cps.is_empty() && index.contains(',') {
            return None;
        }
        waits |= !cps.is_empty();
        items.push((header, pair[1].to_string(), index, cps));
    }
    if !waits {
        return None;
    }
    let base = &req.params();
    // Items run concurrently (as msearch items do), each answer in its place.
    let runs = items.into_iter().map(|(header, body, index, cps)| async move {
        let mut q: Vec<String> = ["rest_total_hits_as_int", "typed_keys"].iter().filter_map(|k| base.get(*k).map(|v| format!("{k}={}", encode_component(v)))).collect();
        for k in ["ignore_unavailable", "allow_no_indices", "expand_wildcards", "allow_partial_search_results", "wait_for_checkpoints_timeout"] {
            if let Some(v) = header.get(k).filter(|_| !cps.is_empty() || k != "wait_for_checkpoints_timeout") {
                q.push(format!("{k}={}", encode_component(&elasticsearch_engine::query::dsl::scalar_string(v))));
            }
        }
        let path = if cps.is_empty() { format!("/{}/_search", encode_component(&index)) } else {
            q.push(format!("wait_for_checkpoints={}", cps.iter().map(i64::to_string).collect::<Vec<_>>().join(",")));
            format!("/{}/_fleet/_fleet_search", encode_component(&index))
        };
        let uri = if q.is_empty() { path } else { format!("{path}?{}", q.join("&")) };
        let prepared=contexts::json_req(req,"POST",&uri,&Value::Null).and_then(|mut sub|{sub.set_body(body.as_bytes())?;Ok(sub)});
        let r = match prepared {
            Err(e)=>RestResp::json(e.status,&e.to_json()),
            Ok(sub)=>if cps.is_empty() {Box::pin(read(node,&sub,&index)).await} else {Box::pin(fleet_search(node,&sub,&index,&cps)).await}
        };
        elasticsearch_engine::source_json::with_member(
            std::str::from_utf8(&r.body).unwrap_or(""), "status", &r.status.to_string(), false,
        ).unwrap_or_else(|| json!({"error": {"type": "illegal_state_exception", "reason": "unreadable response"}, "status": 500}).to_string())
    });
    use futures::StreamExt;
    let responses: Vec<String> = futures::stream::iter(runs).buffered(msearch_concurrency(node, req)).collect().await;
    Some(RestResp::new(200,vec![("content-type".into(), "application/json".into())],format!("{{\"took\":{},\"responses\":[{}]}}", started.elapsed().as_millis(), responses.join(",")).into_bytes()))
}
