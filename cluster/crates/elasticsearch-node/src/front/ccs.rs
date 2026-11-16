//! Cross-cluster search (ES `TransportSearchAction`'s CCS paths).
//!
//! An index expression naming remote clusters (`cluster:index`, wildcard
//! aliases, `-cluster:*` exclusions; `RemoteClusterAware.groupClusterIndices`)
//! is split into this cluster's part and one part per remote cluster. Each
//! remote part runs on the remote cluster over its remote-cluster
//! connection ([`crate::remote_cluster`]), as the calling user (the remote
//! cluster's own role definitions authorize it).
//!
//! **`_search`**: one final reduce on this coordinating node over every
//! cluster's *shard* results. The search body is rewritten once for the
//! shards (`fanout::node_body`: `from` 0, `size` `from + size`, mergeable
//! helper aggregations); this cluster's part and every remote part run it
//! through the per-shard fan-out of their own cluster and answer the
//! unmerged shard results ([`super::ccs_parts`]) with their failed shards.
//! The merge is the ordinary cross-shard merge, so hits, totals,
//! aggregations (sketch states, `shard_size` terms with their error
//! bounds), suggest and profile are reduced exactly as across this
//! cluster's own shards. Remote hits are labelled `alias:index`.
//! `ccs_minimize_roundtrips` (default `true`) keeps ES's semantics where
//! they are visible: one request per remote cluster carries the whole
//! search there, `_clusters` reports per-cluster `took`, and a remote whose
//! every shard was skipped counts as successful only without minimized
//! round trips (`CCSSingleCoordinatorSearchProgressListener`).
//!
//! **Unavailable remotes** (`skip_unavailable`, `true` by default in 9.x):
//! a skippable remote that is not connected, cannot be reached, or fails
//! (index not found included) is reported `skipped` with its failure; a
//! non-skippable one fails the request with the remote's error. Remote
//! shard failures make the remote `partial`.
//!
//! **`_count`, `_terms_enum`**: the clusters' final answers are additive
//! (counts, `_shards`; a sorted union of terms) and merged as such.
//! **`_field_caps`**: per-index capabilities from every cluster
//! (`alias:index`) merged as across this cluster's indices.
//! **`_resolve/index`**: names of remote indices, aliases and data streams
//! are prefixed `alias:`. **Async search** submits the cross-cluster search
//! in the background. **`_msearch`** items are searches of their own.
//!
//! Not supported across clusters here (explicit errors): scroll, point in
//! time, `dfs_query_then_fetch`, retrievers / `rank`, `collapse` with
//! `inner_hits` and kNN that needs ES's DFS round.

use super::*;
use serde_json::Map;
use std::collections::BTreeMap;
use crate::remote_cluster::{self as rc, RemoteRestReq};
use elasticsearch_engine::ccr::remote as eremote;
use elasticsearch_engine::error::EsError;

/// The shard results of a part that matches no index.
pub(crate) fn no_parts() -> Value {
    json!({"_ccs_parts": [], "_ccs_failures": {"total": 0, "successful": 0, "skipped": 0, "failed": 0}, "took": 0})
}

/// One cluster's part of a cross-cluster request.
#[derive(Clone, Debug)]
pub(crate) struct ClusterPart {
    /// `""` for this cluster.
    pub alias: String,
    /// The index expressions it reads.
    pub indices: Vec<String>,
    pub skip_unavailable: bool,
}

/// `RemoteClusterAware.groupClusterIndices` for this node.
pub(crate) fn plan(node: &ClusterNode, targets: &str) -> Result<(Vec<String>, Vec<ClusterPart>), EsError> {
    let (local, remotes) = eremote::group_indices(&node.engine, targets)?;
    let parts = remotes
        .into_iter()
        .map(|(alias, indices)| {
            let skip = eremote::remote_config(&node.engine, &alias).skip_unavailable;
            ClusterPart { alias, indices, skip_unavailable: skip }
        })
        .collect();
    Ok((local, parts))
}

/// The calling user, for the remote cluster's authorization.
pub(crate) async fn identity(node: &ClusterNode, req: &RestReq) -> Option<String> {
    let authn = front_caller(node, req).await?;
    Some(elasticsearch_engine::security::owner::StoredAuthn::of(&authn).encode())
}

/// One cluster's outcome, for `_clusters`.
#[derive(Clone, Debug)]
struct Outcome {
    alias: String,
    indices: String,
    status: &'static str,
    took: Option<u64>,
    timed_out: bool,
    shards: Option<[u64; 4]>,
    failures: Vec<Value>,
}

/// `SearchResponse.Clusters` (`_clusters`) as ES renders it.
fn clusters_json(outcomes: &[Outcome]) -> Value {
    let count = |s: &str| outcomes.iter().filter(|o| o.status == s).count();
    let mut details = Map::new();
    for o in outcomes {
        let mut d = Map::new();
        d.insert("status".into(), json!(o.status));
        d.insert("indices".into(), json!(o.indices));
        if let Some(t) = o.took {
            d.insert("took".into(), json!(t));
        }
        d.insert("timed_out".into(), json!(o.timed_out));
        if let Some([total, ok, skipped, failed]) = o.shards {
            d.insert("_shards".into(), json!({"total": total, "successful": ok, "skipped": skipped, "failed": failed}));
        }
        if !o.failures.is_empty() {
            d.insert("failures".into(), Value::Array(o.failures.clone()));
        }
        details.insert(if o.alias.is_empty() { "(local)".to_string() } else { o.alias.clone() }, Value::Object(d));
    }
    json!({
        "total": outcomes.len(), "successful": count("successful"), "skipped": count("skipped"), "running": count("running"),
        "partial": count("partial"), "failed": count("failed"), "details": details,
    })
}

/// A remote failure as a `_clusters` failure entry (`ShardSearchFailure`).
fn cluster_failure(e: &EsError) -> Value {
    json!({"shard": -1, "index": null, "reason": rc::error_body(e)})
}

/// The error a failed remote request answered (or the transport failure).
fn remote_failure(r: &Result<rc::RemoteRestResp, EsError>) -> Option<EsError> {
    match r {
        Err(e) => Some(e.clone()),
        Ok(resp) if resp.status >= 300 => {
            let v = resp.json().unwrap_or(Value::Null);
            let e = &v["error"];
            let kind = e["type"].as_str().unwrap_or("exception");
            let mut out = EsError::new(resp.status, kind, e["reason"].as_str().unwrap_or_default().to_string());
            if let Some(o) = e.as_object() {
                for (k, x) in o {
                    if !matches!(k.as_str(), "type" | "reason" | "root_cause" | "stack_trace" | "caused_by" | "suppressed") {
                        out = out.with(k, x.clone());
                    }
                }
            }
            if let Some(c) = e.get("caused_by").filter(|c| c.is_object()) {
                out = out.caused_by(EsError::new(resp.status, c["type"].as_str().unwrap_or("exception"), c["reason"].as_str().unwrap_or_default().to_string()));
            }
            Some(out)
        }
        Ok(_) => None,
    }
}

/// `alias:index` for every hit and shard failure of a remote part.
fn label_remote(v: &mut Value, alias: &str) {
    let prefix = |s: &str| if eremote::is_remote_part(s) { s.to_string() } else { format!("{alias}:{s}") };
    for hit in v.pointer_mut("/hits/hits").and_then(Value::as_array_mut).into_iter().flatten() {
        if let Some(i) = hit.get("_index").and_then(Value::as_str).map(prefix) {
            hit["_index"] = json!(i);
        }
        // Inner hits name their own index.
        if let Some(inner) = hit.get_mut("inner_hits").and_then(Value::as_object_mut) {
            for ih in inner.values_mut() {
                for h in ih.pointer_mut("/hits/hits").and_then(Value::as_array_mut).into_iter().flatten() {
                    if let Some(i) = h.get("_index").and_then(Value::as_str).map(prefix) {
                        h["_index"] = json!(i);
                    }
                }
            }
        }
    }
    for f in v.pointer_mut("/_shards/failures").and_then(Value::as_array_mut).into_iter().flatten() {
        if let Some(i) = f.get("index").and_then(Value::as_str).map(prefix) {
            f["index"] = json!(i);
        }
    }
    for f in v.pointer_mut("/failures").and_then(Value::as_array_mut).into_iter().flatten() {
        if let Some(i) = f.get("index").and_then(Value::as_str).map(prefix) {
            f["index"] = json!(i);
        }
    }
    for s in v.pointer_mut("/profile/shards").and_then(Value::as_array_mut).into_iter().flatten() {
        if let Some(i) = s.get("index").and_then(Value::as_str).map(prefix) {
            s["index"] = json!(i);
            s["cluster"] = json!(alias);
        }
    }
}

/// The query string for a cluster's part: everything but the window (in
/// the body) and the coordinator-only parameters.
fn part_query(req: &RestReq) -> String {
    req.query()
        .split('&')
        .filter(|p| !p.is_empty())
        .filter(|p| {
            let k = p.split('=').next().unwrap_or("");
            !matches!(k, "from" | "size" | "ccs_minimize_roundtrips" | "filter_path" | "pretty" | "human" | "error_trace")
        })
        .collect::<Vec<_>>()
        .join("&")
}

fn with_query(path: String, query: &str) -> String {
    if query.is_empty() { path } else { format!("{path}?{query}") }
}

/// Bodies the cross-cluster reduce does not combine (ES runs them with
/// extra coordinated rounds).
fn unsupported(body: &Value, params: &HashMap<String, String>) -> Option<&'static str> {
    // `collapse.inner_hits` and kNN needing DFS run their extra rounds as
    // cross-cluster searches ([`search_on`]).
    // A point in time is each cluster's own (`search_pit`).
    let mut b = body.clone();
    if let Some(o) = b.as_object_mut() {
        o.remove("pit");
    }
    fanout::unsupported_feature(&b, params)
}

fn unsupported_error(what: &str) -> EsError {
    EsError::illegal_argument(format!("{what} is not supported with cross-cluster search"))
}

/// `_search` / `_count` / `_terms_enum` / `POST _async_search` /
/// `_pit` naming remote clusters.
pub(super) async fn read(node: &Arc<ClusterNode>, req: &RestReq, targets: &str) -> RestResp {
    let started = std::time::Instant::now();
    let wildcard = targets.contains(['*', '?']);
    let mrt = req.params().get("ccs_minimize_roundtrips").is_none_or(|v| v != "false");
    let (local, remotes) = match plan(node, targets) {
        Ok(p) => p,
        Err(error) => {
            node.engine.ccs_stats.lock().record(started.elapsed().as_millis() as u64, &[], &[], false, mrt, wildcard);
            return err(error);
        }
    };
    let op = req.segments().last().cloned().unwrap_or_default();
    match op.as_str() {
        // The whole cross-cluster search in the background (its `read`
        // comes back here as `_search`).
        "_async_search" => {
            let holders = Default::default();
            return async_search_ccs(node, req, &holders, None).await;
        }
        "_pit" => return open_pit(node, req, local, remotes).await,
        "_count" | "_terms_enum" => return additive(node, req, &op, local, remotes, started, wildcard, mrt).await,
        _ => {}
    }
    if req.params().contains_key("scroll") {
        return start_scroll(node, req, local, remotes, started, wildcard).await;
    }
    search(node, req, local, remotes, started, wildcard, mrt).await
}

/// Send `uri` (with `body`) to `remote` as the caller; `ensure`: connect first.
async fn remote_call(node: &Arc<ClusterNode>, remote: &ClusterPart, method: &str, uri: String, body: Vec<u8>, identity: Option<String>, parts: bool) -> (Result<rc::RemoteRestResp, EsError>, u64) {
    let t0 = std::time::Instant::now();
    let Some(svc) = rc::service(node) else {
        return (Err(eremote::connection_error(&node.engine, &remote.alias)), 0);
    };
    let r = svc.rest(&remote.alias, RemoteRestReq { method: method.into(), uri, body, identity, parts, ..Default::default() }, !remote.skip_unavailable).await;
    (r, t0.elapsed().as_millis() as u64)
}

/// One round of a cross-cluster search: this cluster's part and every
/// remote part run `shard_body` and answer their shard results (or, with a
/// DFS round, their shards' merged term statistics).
#[allow(clippy::too_many_arguments)]
/// Other-cluster shard count that puts a part above any can_match
/// pre-filter threshold (`ccs_minimize_roundtrips=false`).
const ALWAYS_PREFILTER: usize = 1 << 20;

async fn cluster_round(node: &Arc<ClusterNode>, req: &RestReq, local: &[String], remotes: &[ClusterPart], shard_body: &[u8], query: &str, identity: &Option<String>, dfs: Option<String>, pit: Option<&Map<String, Value>>)
    -> (Option<(RestResp, u64)>, Vec<(Result<rc::RemoteRestResp, EsError>, u64)>) {
    let rank_apply = req.headers.iter().any(|(k, v)| k.eq_ignore_ascii_case(rc::RANK_APPLY_HEADER) && v == "apply");
    // `ccs_minimize_roundtrips=false`: ES asks each remote's search_shards,
    // which always runs can_match; the pre-filter applies whatever the count.
    let always_prefilter = req.params().get("ccs_minimize_roundtrips").is_some_and(|v| v == "false");
    // A cross-cluster point in time: each cluster searches its own.
    let pit_body = |alias: &str| -> Option<Vec<u8>> {
        let id = pit?.get(alias)?.clone();
        let mut b: Value = serde_json::from_slice(shard_body).ok()?;
        b["pit"]["id"] = id;
        serde_json::to_vec(&b).ok()
    };
    let local_part = async {
        if pit.map_or(local.is_empty(), |p| !p.contains_key("")) {
            return None;
        }
        let expr = local.join(",");
        let t0 = std::time::Instant::now();
        let build = || -> Result<RestReq, EsError> {
            let mut r = match pit_body("") {
                Some(b) => req.try_with_body(&b)?,
                None => req.try_with_body(&index_terms_for(&rank_body_for(shard_body, ""), ""))?,
            };
            r.set_method("POST")?;
            r.set_uri(&with_query(if pit.is_some() { "/_search".to_string() } else { format!("/{}/_search", encode_targets(&expr)) }, query))?;
            r.retain_headers(|(k, _)| !k.eq_ignore_ascii_case("accept-encoding"))?;
            r.push_header(rc::CCS_PARTS_HEADER, "1")?;
            if !remotes.is_empty() {
                let others = if always_prefilter { ALWAYS_PREFILTER } else { remotes.len() };
                r.push_header(rc::CCS_OTHER_SHARDS_HEADER, &others.to_string())?;
            }
            if let Some(d) = &dfs {
                r.push_header(rc::CCS_DFS_HEADER, d)?;
            }
            if !r.headers.iter().any(|(h, _)| h.eq_ignore_ascii_case("content-type")) {
                r.push_header("content-type", "application/json")?;
            }
            Ok(r)
        };
        let r = match build() {
            Ok(sub) => Box::pin(super::read(node, &sub, &expr)).await,
            Err(e) => err(e),
        };
        Some((r, t0.elapsed().as_millis() as u64))
    };
    let remote_parts = futures::future::join_all(remotes.iter().map(|remote| {
        let (uri, body) = match pit_body(&remote.alias) {
            Some(b) => (with_query("/_search".to_string(), query), b),
            None => (with_query(format!("/{}/_search", encode_targets(&remote.indices.join(","))), query), index_terms_for(&rank_body_for(shard_body, &remote.alias), &remote.alias)),
        };
        let others = if always_prefilter { ALWAYS_PREFILTER } else { remotes.len() - 1 + usize::from(!local.is_empty() || pit.is_some_and(|p| p.contains_key(""))) };
        let rr = RemoteRestReq { method: "POST".into(), uri, body, identity: identity.clone(), parts: true, dfs: dfs.clone(), rank_apply, other_shards: others as u32 };
        remote_send(node, remote, rr)
    }));
    futures::join!(local_part, remote_parts)
}

/// Send `rr` to `remote`; `ensure`: connect first unless skippable.
async fn remote_send(node: &Arc<ClusterNode>, remote: &ClusterPart, rr: RemoteRestReq) -> (Result<rc::RemoteRestResp, EsError>, u64) {
    let t0 = std::time::Instant::now();
    let Some(svc) = rc::service(node) else {
        return (Err(eremote::connection_error(&node.engine, &remote.alias)), 0);
    };
    let r = svc.rest(&remote.alias, rr, !remote.skip_unavailable).await;
    (r, t0.elapsed().as_millis() as u64)
}

/// A retriever's final body for one cluster: its rank plan keeps only that
/// cluster's documents, named as that cluster names its indices (`""`:
/// this cluster). Other bodies are unchanged.
fn rank_body_for(body: &[u8], alias: &str) -> Vec<u8> {
    let key = elasticsearch_engine::retriever::RANK_PLAN_KEY;
    if !body.windows(key.len()).any(|w| w == key.as_bytes()) {
        return body.to_vec();
    }
    let Ok(mut v) = serde_json::from_slice::<Value>(body) else { return body.to_vec() };
    let own = |name: &str| -> Option<String> {
        match name.split_once(':').filter(|_| eremote::is_remote_part(name)) {
            Some((a, rest)) => (a == alias).then(|| rest.to_string()),
            _ => alias.is_empty().then(|| name.to_string()),
        }
    };
    if let Some(plan) = v.get_mut(key) {
        if let Some(docs) = plan.get_mut("docs").and_then(Value::as_object_mut) {
            let kept: Map<String, Value> = std::mem::take(docs).into_iter().filter_map(|(k, d)| own(&k).map(|n| (n, d))).collect();
            *docs = kept;
        }
        if let Some(order) = plan.get_mut("order").and_then(Value::as_array_mut) {
            let kept: Vec<Value> = std::mem::take(order).into_iter().filter_map(|mut e| {
                let n = own(e.get(0)?.as_str()?)?;
                e[0] = json!(n);
                Some(e)
            }).collect();
            *order = kept;
        }
    }
    serde_json::to_vec(&v).unwrap_or_else(|_| body.to_vec())
}

/// `alias:index` as cluster `alias` (`""`: this cluster) names it: `None`
/// when the name is another cluster's (ES matches `_index` against the
/// shard's cluster-qualified name).
fn own_name(name: &str, alias: &str) -> Option<String> {
    match name.split_once(':').filter(|_| eremote::is_remote_part(name)) {
        Some((a, rest)) => (a == alias).then(|| rest.to_string()),
        _ => alias.is_empty().then(|| name.to_string()),
    }
}

/// A search body for one cluster: `term` / `terms` queries on `_index`
/// match the shard's cluster-qualified name, as ES's `IndexFieldMapper`
/// does (`alias:index` on a remote cluster, the plain name here): names
/// of this cluster are rewritten as it names its indices, others match
/// nothing there. Merged hits are labelled the same way, so a kNN's
/// global top k folded into the query (`fanout::knn_rewrite`) keeps each
/// cluster's own documents.
fn index_terms_for(body: &[u8], alias: &str) -> Vec<u8> {
    if !body.windows(8).any(|w| w == b"\"_index\"") {
        return body.to_vec();
    }
    let Ok(mut v) = serde_json::from_slice::<Value>(body) else { return body.to_vec() };
    const NONE: &str = "\u{0}another-cluster";
    fn walk(v: &mut Value, alias: &str) {
        match v {
            Value::Object(o) => {
                if let Some(t) = o.get_mut("term").and_then(|t| t.get_mut("_index")) {
                    let slot = if t.is_object() { t.get_mut("value") } else { Some(t) };
                    if let Some(s) = slot.filter(|s| s.is_string()) {
                        *s = json!(own_name(s.as_str().unwrap_or_default(), alias).unwrap_or_else(|| NONE.into()));
                    }
                }
                if let Some(Value::Array(names)) = o.get_mut("terms").and_then(|t| t.get_mut("_index")) {
                    for n in names.iter_mut().filter(|n| n.is_string()) {
                        *n = json!(own_name(n.as_str().unwrap_or_default(), alias).unwrap_or_else(|| NONE.into()));
                    }
                }
                for child in o.values_mut() {
                    walk(child, alias);
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|c| walk(c, alias)),
            _ => {}
        }
    }
    walk(&mut v, alias);
    serde_json::to_vec(&v).unwrap_or_else(|_| body.to_vec())
}

#[allow(clippy::too_many_arguments)]
async fn search(node: &Arc<ClusterNode>, req: &RestReq, local: Vec<String>, remotes: Vec<ClusterPart>, started: std::time::Instant, wildcard: bool, mrt: bool) -> RestResp {
    search_on(node, req, local, remotes, started, wildcard, mrt, None).await
}

/// [`search_on`] as an owned, boxed future: the rounds of its expand and
/// DFS kNN phases (a separate function, so the recursive future's `Send`
/// is not judged inside its own body).
#[allow(clippy::too_many_arguments)]
fn search_on_boxed(node: Arc<ClusterNode>, sub: RestReq, local: Vec<String>, remotes: Vec<ClusterPart>, started: std::time::Instant, wildcard: bool, mrt: bool, pit: Option<(String, Map<String, Value>)>) -> futures::future::BoxFuture<'static, RestResp> {
    Box::pin(async move { search_on(&node, &sub, local, remotes, started, wildcard, mrt, pit).await })
}

/// [`search`]; `pit`: a cross-cluster point in time (its wrapped id and
/// each cluster's own id).
#[allow(clippy::too_many_arguments)]
async fn search_on(node: &Arc<ClusterNode>, req: &RestReq, local: Vec<String>, remotes: Vec<ClusterPart>, started: std::time::Instant, wildcard: bool, mrt: bool, pit: Option<(String, Map<String, Value>)>) -> RestResp {
    let params = req.params();
    let empty = json!({});
    let body: &Value = req.json_ref().unwrap_or(&empty);
    let aliases: Vec<String> = remotes.iter().map(|r| r.alias.clone()).collect();
    let record = |ok: bool, skipped: &[String]| {
        node.engine.ccs_stats.lock().record(started.elapsed().as_millis() as u64, &aliases, skipped, ok, mrt, wildcard);
    };
    // Retrievers and rank: the plan phase runs on this node, its child
    // searches are cross-cluster searches of their own; the final search
    // (the rank plan applied) comes back here.
    let applying = req.headers.iter().any(|(k, v)| k.eq_ignore_ascii_case(rc::RANK_APPLY_HEADER) && v == "apply");
    if !applying && is_rank_search(body, &params) {
        record(true, &[]);
        return Box::pin(rank_search(node, req)).await;
    }
    // ES's expand phase (`collapse.inner_hits`) and DFS kNN phase across
    // clusters: their rounds are cross-cluster searches over the same
    // clusters (ES's `ExpandSearchPhase` / `DfsQueryPhase` keep the
    // request's remote indices); the kNN's global top k folds into the
    // query by `_index` (each cluster keeps its own documents, see
    // [`index_terms_for`]).
    if !fanout::collapse_inner_hits(body).is_empty() || fanout::knn_needs_dfs(body, &params) {
        let run = |sub: RestReq| search_on_boxed(node.clone(), sub, local.clone(), remotes.clone(), started, wildcard, mrt, pit.clone());
        if let Some(r) = super::collapse_expand(req, &run).await {
            return r;
        }
        match super::knn_dfs(req, &run).await {
            Err(r) => return r,
            Ok(Some(rewritten)) => return match super::force_dfs(rewritten) {
                Ok(rewritten) => run(rewritten).await,
                Err(e) => err(e),
            },
            Ok(None) => {}
        }
    }
    if let Some(what) = unsupported(body, &params) {
        record(false, &[]);
        return err(unsupported_error(what));
    }
    let (from, size) = fanout::from_size(body, &params);
    let shard_body = elasticsearch_engine::source_json::search_body(&req.body, &fanout::node_body(body, from, size));
    let query = part_query(req);
    let identity = identity(node, req).await;
    // ES's DFS phase across clusters: every cluster's shards' term
    // statistics, merged, then every cluster scores with the global ones.
    let mut dfs: Option<String> = None;
    if params.get("search_type").map(String::as_str) == Some("dfs_query_then_fetch") {
        let (local_stats, remote_stats) = cluster_round(node, req, &local, &remotes, &shard_body, &query, &identity, Some("collect".into()), pit.as_ref().map(|p| &p.1)).await;
        let mut stats: Vec<elasticsearch_http::DfsStats> = Vec::new();
        if let Some((r, _)) = local_stats {
            if !r.is_success() {
                record(false, &[]);
                return r;
            }
            stats.extend(serde_json::from_slice::<elasticsearch_http::DfsStats>(&r.body).ok());
        }
        for (remote, (r, _)) in remotes.iter().zip(remote_stats) {
            match remote_failure(&r) {
                Some(_) if remote.skip_unavailable => {}
                Some(e) => {
                    record(false, &[]);
                    return err(e);
                }
                None => stats.extend(r.ok().and_then(|r| serde_json::from_slice::<elasticsearch_http::DfsStats>(&r.body).ok())),
            }
        }
        let merged = elasticsearch_http::merge_dfs(&stats);
        dfs = Some(format!("use:{}", eremote::hex_encode(&serde_json::to_vec(&merged).unwrap_or_default())));
    }
    let (local_result, remote_results) = cluster_round(node, req, &local, &remotes, &shard_body, &query, &identity, dfs, pit.as_ref().map(|p| &p.1)).await;

    let mut sources = elasticsearch_engine::source_json::ResponseSources::default();
    let memory = elasticsearch_engine::memory::RequestMemory::new();
    let mut parts: Vec<Value> = Vec::new();
    let mut failures: Vec<Value> = Vec::new();
    let mut failed_shards = 0u64;
    let mut failed_total = 0u64;
    let mut outcomes: Vec<Outcome> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    // Shard results and failures of one cluster's answer.
    let mut take = |text: &[u8], alias: &str, took: u64, skippable: bool| -> Result<Outcome, EsError> {
        let mut v = parse_search_part(&mut sources, text, &memory)?;
        let mut sums = [0u64; 4];
        let mut timed_out = false;
        let mut cluster_parts = match v.get_mut("_ccs_parts").map(Value::take) {
            Some(Value::Array(a)) => a,
            _ => Vec::new(),
        };
        for p in cluster_parts.iter_mut() {
            if !alias.is_empty() {
                label_remote(p, alias);
            }
            for (i, k) in ["total", "successful", "skipped", "failed"].iter().enumerate() {
                sums[i] += p["_shards"][k].as_u64().unwrap_or(0);
            }
            timed_out |= p["timed_out"].as_bool().unwrap_or(false);
        }
        let mut f = v.get("_ccs_failures").cloned().unwrap_or(Value::Null);
        if !alias.is_empty() {
            label_remote(&mut f, alias);
        }
        let n_failed = f["failed"].as_u64().unwrap_or(0);
        sums[0] += f["total"].as_u64().unwrap_or(0);
        sums[3] += n_failed;
        let cluster_failures: Vec<Value> = f["failures"].as_array().cloned().unwrap_or_default();
        failed_shards += n_failed;
        failed_total += f["total"].as_u64().unwrap_or(0);
        failures.extend(cluster_failures.iter().cloned());
        parts.extend(cluster_parts);
        let status = if sums[0] > 0 && sums[3] >= sums[0] {
            if skippable { "skipped" } else { "failed" }
        } else if timed_out || sums[3] > 0 {
            "partial"
        } else {
            "successful"
        };
        Ok(Outcome { alias: alias.to_string(), indices: String::new(), status, took: Some(took), timed_out, shards: Some(sums), failures: cluster_failures })
    };

    if let Some((r, took)) = local_result {
        if !r.is_success() {
            record(false, &[]);
            return r;
        }
        match take(&r.body, "", took, false) {
            Ok(mut o) => {
                o.indices = local.join(",");
                outcomes.push(o);
            }
            Err(e) => {
                record(false, &[]);
                return err(e);
            }
        }
    }
    for (remote, (r, took)) in remotes.iter().zip(remote_results) {
        let indices = remote.indices.join(",");
        match remote_failure(&r) {
            Some(e) => {
                if !remote.skip_unavailable {
                    record(false, &[]);
                    tracing::debug!(alias = %remote.alias, "error while communicating with remote cluster [{}]: {e}", remote.alias);
                    return err(e);
                }
                skipped.push(remote.alias.clone());
                outcomes.push(Outcome { alias: remote.alias.clone(), indices, status: "skipped", took: None, timed_out: false, shards: None, failures: vec![cluster_failure(&e)] });
            }
            None => {
                let resp = r.expect("checked");
                match take(&resp.body, &remote.alias, took, remote.skip_unavailable) {
                    Ok(mut o) => {
                        o.indices = indices;
                        if o.status == "skipped" {
                            skipped.push(remote.alias.clone());
                        }
                        if o.status == "failed" {
                            record(false, &[]);
                            let reason = o.failures.first().map(|f| f["reason"].clone()).unwrap_or(Value::Null);
                            return RestResp::json(503, &json!({"error": {"root_cause": [reason.clone()], "type": "search_phase_execution_exception", "reason": "all shards failed", "phase": "query", "grouped": true, "failed_shards": o.failures}, "status": 503}));
                        }
                        outcomes.push(o);
                    }
                    Err(e) => {
                        if !remote.skip_unavailable {
                            record(false, &[]);
                            return err(e);
                        }
                        skipped.push(remote.alias.clone());
                        outcomes.push(Outcome { alias: remote.alias.clone(), indices, status: "skipped", took: None, timed_out: false, shards: None, failures: vec![cluster_failure(&e)] });
                    }
                }
            }
        }
    }
    drop(take);
    // `allow_partial_search_results=false`: a failed shard anywhere fails the search.
    if failed_shards > 0 && !copies::allow_partial(node, &params) {
        record(false, &skipped);
        return RestResp::json(503, &json!({"error": {"root_cause": failures.iter().map(|f| f["reason"].clone()).collect::<Vec<_>>(), "type": "search_phase_execution_exception",
            "reason": "Partial shards failure", "phase": "query", "grouped": true, "failed_shards": failures}, "status": 503}));
    }
    let merged = fanout::merge_search_with_source(&parts, body, &params, node.engine.max_buckets(), &req.body);
    if let Some(e) = fanout::take_agg_breaking() {
        record(false, &skipped);
        return RestResp::json(e.status, &e.to_json());
    }
    if let Some(e) = fanout::take_agg_error() {
        record(false, &skipped);
        return RestResp::json(e.status, &e.to_json());
    }
    let mut merged = merged;
    if let Some(s) = merged.get_mut("_shards").and_then(Value::as_object_mut) {
        let total = s.get("total").and_then(Value::as_u64).unwrap_or(0) + failed_total;
        let failed = s.get("failed").and_then(Value::as_u64).unwrap_or(0) + failed_shards;
        s.insert("total".into(), json!(total));
        s.insert("failed".into(), json!(failed));
        if !failures.is_empty() {
            let mut list: Vec<Value> = s.remove("failures").and_then(|f| f.as_array().cloned()).unwrap_or_default();
            list.extend(failures.iter().cloned());
            s.insert("failures".into(), Value::Array(list));
        }
    }
    if merged.get("took").is_some() || merged.is_object() {
        merged["took"] = json!(started.elapsed().as_millis() as u64);
    }
    // No shard answered (every remote skipped): still ES's empty `total`.
    if merged["hits"].is_object() && merged["hits"].get("total").is_none() && body.get("track_total_hits") != Some(&Value::Bool(false)) {
        let int = params.get("rest_total_hits_as_int").is_some_and(|v| v.is_empty() || v == "true");
        merged["hits"]["total"] = if int { json!(0) } else { json!({"value": 0, "relation": "eq"}) };
    }
    // `_clusters` right after `_shards`, as ES orders it.
    let clusters = clusters_json(&outcomes);
    let mut ordered = Map::new();
    if let Some((wrapped, _)) = &pit {
        ordered.insert("pit_id".into(), json!(wrapped));
    }
    for (k, v) in merged.as_object().cloned().unwrap_or_default() {
        let is_shards = k == "_shards";
        ordered.insert(k, v);
        if is_shards {
            ordered.insert("_clusters".into(), clusters.clone());
        }
    }
    if !ordered.contains_key("_clusters") {
        ordered.insert("_clusters".into(), clusters);
    }
    record(true, &skipped);
    let rendered = sources.render(&Value::Object(ordered));
    let mut raw = RestResp::json(200, &Value::Null);
    raw.replace_body_vec(rendered);
    crate::rest::util::postprocess(req, raw)
}

/// `_count` / `_terms_enum`: every cluster's final answer, merged.
#[allow(clippy::too_many_arguments)]
async fn additive(node: &Arc<ClusterNode>, req: &RestReq, op: &str, local: Vec<String>, remotes: Vec<ClusterPart>, started: std::time::Instant, wildcard: bool, mrt: bool) -> RestResp {
    let aliases: Vec<String> = remotes.iter().map(|r| r.alias.clone()).collect();
    let query = part_query(req);
    let identity = identity(node, req).await;
    let body = req.body.to_vec();
    let method = req.method.clone();
    let local_part = async {
        if local.is_empty() {
            return None;
        }
        let expr = local.join(",");
        let mut sub = req.clone();
        if let Err(e) = sub.set_uri(&with_query(format!("/{}/{op}", encode_targets(&expr)), &query)) {
            return Some(err(e));
        }
        Some(Box::pin(super::read(node, &sub, &expr)).await)
    };
    let remote_parts = futures::future::join_all(remotes.iter().map(|remote| {
        let uri = with_query(format!("/{}/{op}", encode_targets(&remote.indices.join(","))), &query);
        remote_call(node, remote, &method, uri, body.clone(), identity.clone(), false)
    }));
    let (local_result, remote_results) = futures::join!(local_part, remote_parts);
    let mut answers: Vec<Value> = Vec::new();
    let mut skipped = Vec::new();
    if let Some(r) = local_result {
        if !r.is_success() {
            return r;
        }
        answers.push(r.body_json().unwrap_or(Value::Null));
    }
    for (remote, (r, _)) in remotes.iter().zip(remote_results) {
        match remote_failure(&r) {
            Some(_) if remote.skip_unavailable => skipped.push(remote.alias.clone()),
            Some(e) => {
                node.engine.ccs_stats.lock().record(started.elapsed().as_millis() as u64, &aliases, &[], false, mrt, wildcard);
                return err(e);
            }
            None => answers.push(r.ok().and_then(|r| r.json()).unwrap_or(Value::Null)),
        }
    }
    let params = req.params();
    let merged = if op == "_count" {
        fanout::merge_count(&answers)
    } else {
        merge_terms_enum(&answers, req.json_ref().unwrap_or(&Value::Null), &params)
    };
    node.engine.ccs_stats.lock().record(started.elapsed().as_millis() as u64, &aliases, &skipped, true, mrt, wildcard);
    crate::rest::util::postprocess(req, RestResp::json(200, &merged))
}

// ------------------------------------------------------------- field caps

/// `_field_caps` per concrete index of `expr` on this cluster:
/// `(index, response)`.
pub(crate) async fn field_caps_parts(node: &Arc<ClusterNode>, req: &RestReq, expr: &str) -> Result<Vec<(String, Value)>, RestResp> {
    let opts = ResolveOpts::from_params(&req.params());
    let engine = node.engine.clone();
    let e2 = expr.to_string();
    let names: Vec<String> = match tokio::task::spawn_blocking(move || engine.resolve(&e2, opts).map(|t| t.iter().map(|t| t.svc.name.clone()).collect::<BTreeSet<String>>())).await {
        Ok(Ok(n)) => n.into_iter().collect(),
        Ok(Err(e)) => return Err(err(e)),
        Err(e) => return Err(err(EsError::internal(e.to_string()))),
    };
    let query: String = req.query().split('&').filter(|p| !p.is_empty() && !p.starts_with("index_filter")).collect::<Vec<_>>().join("&");
    let mut subs = Vec::with_capacity(names.len());
    for n in &names {
        let mut r = req.clone();
        if let Err(e) = r.retain_headers(|(k, _)| !k.eq_ignore_ascii_case(rc::CCS_PARTS_HEADER)) {
            return Err(err(e));
        }
        if let Err(e) = r.set_uri(&with_query(format!("/{}/_field_caps", encode_component(n)), &query)) {
            return Err(err(e));
        }
        subs.push(r);
    }
    let resps = futures::future::join_all(subs.iter().zip(&names).map(|(r, n)| read_route(node, r, n))).await;
    let mut parts = Vec::with_capacity(names.len());
    for (n, r) in names.iter().zip(resps) {
        if !r.is_success() {
            return Err(r);
        }
        parts.push((n.clone(), r.body_json().unwrap_or(Value::Null)));
    }
    Ok(parts)
}

/// `_field_caps` naming remote clusters.
pub(crate) async fn field_caps(node: &Arc<ClusterNode>, req: &RestReq, targets: &str) -> RestResp {
    let (local, remotes) = match plan(node, targets) {
        Ok(p) => p,
        Err(e) => return err(e),
    };
    let identity = identity(node, req).await;
    let query = req.query().to_string();
    let body = req.body.to_vec();
    let method = req.method.clone();
    let local_part = async {
        if local.is_empty() {
            return None;
        }
        Some(Box::pin(field_caps_parts(node, req, &local.join(","))).await)
    };
    let remote_parts = futures::future::join_all(remotes.iter().map(|remote| {
        let uri = with_query(format!("/{}/_field_caps", encode_targets(&remote.indices.join(","))), &query);
        remote_call(node, remote, &method, uri, body.clone(), identity.clone(), true)
    }));
    let (local_result, remote_results) = futures::join!(local_part, remote_parts);
    let mut parts: Vec<(String, Value)> = Vec::new();
    let mut failures: Vec<Value> = Vec::new();
    if let Some(r) = local_result {
        match r {
            Ok(p) => parts.extend(p),
            Err(r) => return r,
        }
    }
    for (remote, (r, _)) in remotes.iter().zip(remote_results) {
        match remote_failure(&r) {
            Some(e) if remote.skip_unavailable => {
                let indices: Vec<String> = remote.indices.iter().map(|i| format!("{}:{i}", remote.alias)).collect();
                failures.push(json!({"indices": indices, "failure": rc::error_body(&e)}));
            }
            Some(e) => return err(e),
            None => {
                let v = r.ok().and_then(|r| r.json()).unwrap_or(Value::Null);
                for p in v["_ccs_parts"].as_array().into_iter().flatten() {
                    if let (Some(i), Some(resp)) = (p.get(0).and_then(Value::as_str), p.get(1)) {
                        // The answer lists its index under the remote name.
                        let mut resp = resp.clone();
                        for x in resp.get_mut("indices").and_then(Value::as_array_mut).into_iter().flatten() {
                            if let Some(n) = x.as_str().map(|n| format!("{}:{n}", remote.alias)) {
                                *x = json!(n);
                            }
                        }
                        parts.push((format!("{}:{i}", remote.alias), resp));
                    }
                }
            }
        }
    }
    let include_unmapped = req.params().get("include_unmapped").map(|v| v == "true" || v.is_empty()).unwrap_or(false);
    let mut merged = fanout::merge_field_caps(&parts, include_unmapped);
    if !failures.is_empty() {
        merged["failed_indices"] = json!(failures.iter().map(|f| f["indices"].as_array().map_or(0, Vec::len)).sum::<usize>());
        merged["failures"] = Value::Array(failures);
    }
    crate::rest::util::postprocess(req, RestResp::json(200, &merged))
}

/// The per-index field caps of this cluster's part of a coordinating
/// cluster's request ([`super::ccs_parts`]).
pub(crate) async fn field_caps_for_remote(node: &Arc<ClusterNode>, req: &RestReq, expr: &str) -> RestResp {
    match field_caps_parts(node, req, expr).await {
        Ok(parts) => RestResp::json(200, &json!({"_ccs_parts": parts.into_iter().map(|(i, v)| json!([i, v])).collect::<Vec<_>>()})),
        Err(r) => r,
    }
}

// ---------------------------------------------------------- resolve index

/// `GET /_resolve/index/{expr}` naming remote clusters: every cluster's
/// resolution, remote names prefixed `alias:`.
pub(crate) async fn resolve_index(node: &Arc<ClusterNode>, req: &RestReq, expr: &str) -> RestResp {
    let (local, remotes) = match plan(node, expr) {
        Ok(p) => p,
        Err(e) => return err(e),
    };
    let identity = identity(node, req).await;
    let query = req.query().to_string();
    let mut out = json!({"indices": [], "aliases": [], "data_streams": []});
    let push = |out: &mut Value, v: &Value, alias: Option<&str>| {
        for k in ["indices", "aliases", "data_streams"] {
            for e in v[k].as_array().into_iter().flatten() {
                let mut e = e.clone();
                if let (Some(a), Some(n)) = (alias, e.get("name").and_then(Value::as_str).map(str::to_string)) {
                    e["name"] = json!(format!("{a}:{n}"));
                }
                if let Some(arr) = out[k].as_array_mut() {
                    arr.push(e);
                }
            }
        }
    };
    if !local.is_empty() {
        let mut sub = req.clone();
        if let Err(e) = sub.set_uri(&with_query(format!("/_resolve/index/{}", encode_targets(&local.join(","))), &query)) {
            return err(e);
        }
        let r = local_engine(node, &sub).await;
        if !r.is_success() {
            return r;
        }
        push(&mut out, &r.body_json().unwrap_or(Value::Null), None);
    }
    let results = futures::future::join_all(remotes.iter().map(|remote| {
        let uri = with_query(format!("/_resolve/index/{}", encode_targets(&remote.indices.join(","))), &query);
        remote_call(node, remote, "GET", uri, Vec::new(), identity.clone(), false)
    }))
    .await;
    for (remote, (r, _)) in remotes.iter().zip(results) {
        match remote_failure(&r) {
            Some(_) if remote.skip_unavailable => {}
            Some(e) => return err(e),
            None => push(&mut out, &r.ok().and_then(|r| r.json()).unwrap_or(Value::Null), Some(&remote.alias)),
        }
    }
    for k in ["indices", "aliases", "data_streams"] {
        if let Some(arr) = out[k].as_array_mut() {
            arr.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        }
    }
    crate::rest::util::postprocess(req, RestResp::json(200, &out))
}

// --------------------------------------------------------- resolve cluster

/// `_resolve/cluster` entries of the remote clusters an expression names
/// (every configured one for the index-less form), as ES 9 reports them:
/// `connected`, `skip_unavailable`, and for a reachable cluster
/// `matching_indices` (with an expression) and `version`, else `error`.
pub(crate) async fn resolve_cluster_remotes(node: &Arc<ClusterNode>, req: &RestReq, expr: Option<&str>) -> Result<(Option<String>, Map<String, Value>), EsError> {
    let (local, aliases): (Vec<String>, Vec<(String, Vec<String>)>) = match expr {
        Some(expr) => {
            let (local, remotes) = eremote::group_indices(&node.engine, expr)?;
            (local, remotes.into_iter().collect())
        }
        None => (Vec::new(), eremote::remote_aliases(&node.engine).into_iter().map(|a| (a, Vec::new())).collect()),
    };
    let identity = identity(node, req).await;
    let svc = rc::service(node);
    let mut out = Map::new();
    let checks = futures::future::join_all(aliases.iter().map(|(alias, indices)| {
        let svc = svc.clone();
        let identity = identity.clone();
        async move {
            let skip = eremote::remote_config(&node.engine, alias).skip_unavailable;
            let Some(svc) = svc else {
                return (alias.clone(), json!({"connected": false, "skip_unavailable": skip}));
            };
            if let Err(e) = svc.ensure_connected(alias).await {
                return (alias.clone(), json!({"connected": false, "skip_unavailable": skip, "error": e.reason}));
            }
            let version = svc.cluster(alias).and_then(|c| c.version()).unwrap_or_default();
            let version_json = json!({"number": version, "build_flavor": elasticsearch_engine::version::BUILD_FLAVOR,
                "minimum_wire_compatibility_version": elasticsearch_engine::version::MIN_WIRE_COMPAT,
                "minimum_index_compatibility_version": elasticsearch_engine::version::MIN_INDEX_COMPAT});
            if indices.is_empty() {
                return (alias.clone(), json!({"connected": true, "skip_unavailable": skip, "version": version_json}));
            }
            let uri = format!("/_resolve/cluster/{}", encode_targets(&indices.join(",")));
            match svc.rest(alias, RemoteRestReq { method: "GET".into(), uri, body: Vec::new(), identity, parts: false, ..Default::default() }, false).await {
                Ok(r) if r.status < 300 => {
                    let v = r.json().unwrap_or(Value::Null);
                    let mut entry = v.get("(local)").cloned().unwrap_or(json!({}));
                    entry["connected"] = json!(true);
                    entry["skip_unavailable"] = json!(skip);
                    (alias.clone(), entry)
                }
                Ok(r) => {
                    let v = r.json().unwrap_or(Value::Null);
                    (alias.clone(), json!({"connected": true, "skip_unavailable": skip, "error": v.pointer("/error/reason").cloned().unwrap_or(Value::Null)}))
                }
                Err(e) => (alias.clone(), json!({"connected": false, "skip_unavailable": skip, "error": e.reason})),
            }
        }
    }))
    .await;
    for (alias, entry) in checks {
        out.insert(alias, entry);
    }
    Ok(((!local.is_empty()).then(|| local.join(",")), out))
}

// ------------------------------------------------------------------ scroll

/// Prefix of a cross-cluster scroll id's inner part (kept by the
/// coordinating node, wrapped with its id).
pub(crate) const SCROLL_PREFIX: &str = "~x";
/// Prefix of a cross-cluster point in time's inner part.
pub(crate) const PIT_PREFIX: &str = eremote::CCS_PIT_PREFIX;

/// One cluster's stream of a cross-cluster scroll.
struct ScrollCursor {
    alias: String,
    scroll_id: Option<String>,
    buffer: std::collections::VecDeque<Value>,
    exhausted: bool,
}

/// A cross-cluster scroll (ES runs it without minimized round trips): each
/// cluster scrolls its own part; pages are merged in the request's sort
/// order, hits a page did not take stay buffered for the next one.
struct CcsScroll {
    owner: Option<String>,
    identity: Option<String>,
    body: Value,
    params: HashMap<String, String>,
    size: usize,
    clusters: Vec<ScrollCursor>,
    total: Value,
    shards: [u64; 4],
    expires: std::time::Instant,
    keep_alive: String,
}

type ScrollTable = parking_lot::Mutex<HashMap<String, Arc<tokio::sync::Mutex<CcsScroll>>>>;

fn scrolls() -> &'static ScrollTable {
    static T: std::sync::OnceLock<ScrollTable> = std::sync::OnceLock::new();
    T.get_or_init(Default::default)
}

/// The owner of a cross-cluster scroll: the whole authentication, as the
/// engine keys its own scrolls (`search_api::principal`): authentication
/// type, user, realm name and type and API key id of the authenticating
/// subject, then the run-as target's. Two API keys of one user (whose realm
/// is always `_es_api_key`), or same-named users of two realms, never share
/// a scroll (ES `ensureAuthenticatedUserIsSame`).
fn caller_key(a: &Option<elasticsearch_engine::security::Authn>) -> Option<String> {
    fn id(u: &elasticsearch_engine::security::Authn) -> String {
        format!("{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}", u.auth_type, u.username, u.realm_name, u.realm_type, u.api_key_id().unwrap_or(""))
    }
    a.as_ref().map(|a| match &a.run_as {
        Some(target) => format!("{}\u{1e}{}", id(a), id(target)),
        None => id(a),
    })
}

fn keep_alive_of(s: &str) -> Duration {
    Duration::from_millis(elasticsearch_engine::date::parse_time_value(s).unwrap_or(60_000).max(1) as u64)
}

fn context_missing(id: &str) -> RestResp {
    RestResp::json(404, &json!({"error": {"root_cause": [{"type": "search_context_missing_exception", "reason": format!("No search context found for id [{id}]")}],
        "type": "search_phase_execution_exception", "reason": "all shards failed", "phase": "query", "grouped": true,
        "failed_shards": [{"shard": -1, "index": null, "reason": {"type": "search_context_missing_exception", "reason": format!("No search context found for id [{id}]")}}],
        "caused_by": {"type": "search_context_missing_exception", "reason": format!("No search context found for id [{id}]")}}, "status": 404}))
}

/// Add a page of one cluster to its cursor (remote hits labelled).
fn absorb_page(c: &mut ScrollCursor, mut page: Value) {
    if !c.alias.is_empty() {
        label_remote(&mut page, &c.alias);
    }
    if let Some(id) = page.get("_scroll_id").and_then(Value::as_str) {
        c.scroll_id = Some(id.to_string());
    }
    let hits = page.pointer_mut("/hits/hits").map(Value::take).and_then(|h| match h {
        Value::Array(a) => Some(a),
        _ => None,
    }).unwrap_or_default();
    if hits.is_empty() {
        c.exhausted = true;
    }
    c.buffer.extend(hits);
}

/// Up to `size` hits in sort order, refilling buffers that run dry.
async fn next_page(node: &Arc<ClusterNode>, req: &RestReq, st: &mut CcsScroll) -> Result<Vec<Value>, EsError> {
    let keys = fanout::sort_keys(&st.body, &st.params);
    let mut out = Vec::new();
    loop {
        // Refill every live stream with an empty buffer first: its next hit
        // may sort before every buffered one.
        for i in 0..st.clusters.len() {
            if st.clusters[i].buffer.is_empty() && !st.clusters[i].exhausted {
                let page = continue_cluster(node, req, st, i).await?;
                absorb_page(&mut st.clusters[i], page);
            }
        }
        if out.len() >= st.size {
            break;
        }
        let best = (0..st.clusters.len()).filter(|i| !st.clusters[*i].buffer.is_empty()).min_by(|a, b| {
            let (x, y) = (&st.clusters[*a].buffer[0], &st.clusters[*b].buffer[0]);
            fanout::hit_order(x, y, keys.as_deref()).then(a.cmp(b))
        });
        let Some(i) = best else { break };
        if let Some(h) = st.clusters[i].buffer.pop_front() {
            out.push(h);
        }
    }
    Ok(out)
}

/// The next page of cluster `i`'s scroll.
async fn continue_cluster(node: &Arc<ClusterNode>, req: &RestReq, st: &CcsScroll, i: usize) -> Result<Value, EsError> {
    let c = &st.clusters[i];
    let Some(id) = c.scroll_id.clone() else { return Ok(json!({"hits": {"hits": []}})) };
    let body = json!({"scroll_id": id, "scroll": st.keep_alive});
    let rest_int = st.params.get("rest_total_hits_as_int").map(|v| format!("?rest_total_hits_as_int={v}")).unwrap_or_default();
    if c.alias.is_empty() {
        let mut sub = req.clone();
        sub.set_method("POST")?;
        sub.set_uri(&format!("/_search/scroll{rest_int}"))?;
        sub.set_json(&body)?;
        let r = Box::pin(context(node, &sub, ContextKind::Scroll)).await;
        if !r.is_success() {
            return Err(EsError::from_error_json(r.status, &r.body_json().unwrap_or(Value::Null)["error"]));
        }
        return Ok(r.body_json().unwrap_or(Value::Null));
    }
    let svc = rc::service(node).ok_or_else(|| eremote::connection_error(&node.engine, &c.alias))?;
    let r = svc.rest(&c.alias, RemoteRestReq { method: "POST".into(), uri: format!("/_search/scroll{rest_int}"), body: serde_json::to_vec(&body).unwrap_or_default(), identity: st.identity.clone(), parts: false, ..Default::default() }, true).await;
    if let Some(e) = remote_failure(&r) {
        return Err(e);
    }
    Ok(r.ok().and_then(|r| r.json()).unwrap_or(Value::Null))
}

fn scroll_response(id: &str, took: u64, shards: [u64; 4], total: &Value, hits: Vec<Value>) -> Value {
    let max_score = hits.iter().filter_map(|h| h["_score"].as_f64()).fold(None, |m: Option<f64>, s| Some(m.map_or(s, |m| m.max(s))));
    json!({
        "_scroll_id": id, "took": took, "timed_out": false,
        "_shards": {"total": shards[0], "successful": shards[1], "skipped": shards[2], "failed": shards[3]},
        "hits": {"total": total, "max_score": max_score, "hits": hits},
    })
}

/// `_search?scroll` naming remote clusters.
async fn start_scroll(node: &Arc<ClusterNode>, req: &RestReq, local: Vec<String>, remotes: Vec<ClusterPart>, started: std::time::Instant, wildcard: bool) -> RestResp {
    let params = req.params();
    let body = req.json().unwrap_or(json!({}));
    let aliases: Vec<String> = remotes.iter().map(|r| r.alias.clone()).collect();
    let (_, size) = fanout::from_size(&body, &params);
    let caller = front_caller(node, req).await;
    let identity = caller.as_ref().map(|a| elasticsearch_engine::security::owner::StoredAuthn::of(a).encode());
    let keep_alive = params.get("scroll").cloned().unwrap_or_else(|| "1m".into());
    let query: String = req.query().split('&').filter(|p| !p.is_empty() && !matches!(p.split('=').next().unwrap_or(""), "ccs_minimize_roundtrips" | "filter_path" | "pretty" | "human")).collect::<Vec<_>>().join("&");
    let mut st = CcsScroll { owner: caller_key(&caller), identity: identity.clone(), body: body.clone(), params: params.clone(), size, clusters: Vec::new(),
        total: Value::Null, shards: [0; 4], expires: std::time::Instant::now() + keep_alive_of(&keep_alive), keep_alive };
    let mut outcomes = Vec::new();
    let add_total = |total: &mut Value, t: &Value| {
        *total = match (total.take(), t) {
            (Value::Null, t) => t.clone(),
            (Value::Number(a), Value::Number(b)) => json!(a.as_u64().unwrap_or(0) + b.as_u64().unwrap_or(0)),
            (Value::Object(mut a), Value::Object(b)) => {
                let v = a.get("value").and_then(Value::as_u64).unwrap_or(0) + b.get("value").and_then(Value::as_u64).unwrap_or(0);
                a.insert("value".into(), json!(v));
                if b.get("relation") == Some(&json!("gte")) {
                    a.insert("relation".into(), json!("gte"));
                }
                Value::Object(a)
            }
            (a, _) => a,
        };
    };
    if !local.is_empty() {
        let expr = local.join(",");
        let t0 = std::time::Instant::now();
        let mut sub = req.clone();
        if let Err(e) = sub.set_uri(&with_query(format!("/{}/_search", encode_targets(&expr)), &query)) {
            return err(e);
        }
        let r = Box::pin(super::read(node, &sub, &expr)).await;
        if !r.is_success() {
            return r;
        }
        let page = r.body_json().unwrap_or(Value::Null);
        let sh = &page["_shards"];
        let shards = [sh["total"].as_u64().unwrap_or(0), sh["successful"].as_u64().unwrap_or(0), sh["skipped"].as_u64().unwrap_or(0), sh["failed"].as_u64().unwrap_or(0)];
        for (i, n) in shards.iter().enumerate() {
            st.shards[i] += n;
        }
        add_total(&mut st.total, &page["hits"]["total"]);
        outcomes.push(Outcome { alias: String::new(), indices: expr, status: if shards[3] > 0 { "partial" } else { "successful" }, took: Some(t0.elapsed().as_millis() as u64), timed_out: false, shards: Some(shards), failures: Vec::new() });
        let mut c = ScrollCursor { alias: String::new(), scroll_id: None, buffer: Default::default(), exhausted: false };
        absorb_page(&mut c, page);
        st.clusters.push(c);
    }
    let results = futures::future::join_all(remotes.iter().map(|remote| {
        let uri = with_query(format!("/{}/_search", encode_targets(&remote.indices.join(","))), &query);
        remote_call(node, remote, "POST", uri, req.body.to_vec(), identity.clone(), false)
    }))
    .await;
    let mut skipped = Vec::new();
    for (remote, (r, took)) in remotes.iter().zip(results) {
        let indices = remote.indices.join(",");
        if let Some(e) = remote_failure(&r) {
            if !remote.skip_unavailable {
                node.engine.ccs_stats.lock().record(started.elapsed().as_millis() as u64, &aliases, &[], false, false, wildcard);
                return err(e);
            }
            skipped.push(remote.alias.clone());
            outcomes.push(Outcome { alias: remote.alias.clone(), indices, status: "skipped", took: None, timed_out: false, shards: None, failures: vec![cluster_failure(&e)] });
            continue;
        }
        let page = r.ok().and_then(|r| r.json()).unwrap_or(Value::Null);
        let sh = &page["_shards"];
        let shards = [sh["total"].as_u64().unwrap_or(0), sh["successful"].as_u64().unwrap_or(0), sh["skipped"].as_u64().unwrap_or(0), sh["failed"].as_u64().unwrap_or(0)];
        for (i, n) in shards.iter().enumerate() {
            st.shards[i] += n;
        }
        add_total(&mut st.total, &page["hits"]["total"]);
        outcomes.push(Outcome { alias: remote.alias.clone(), indices, status: if shards[3] > 0 { "partial" } else { "successful" }, took: Some(took), timed_out: false, shards: Some(shards), failures: Vec::new() });
        let mut c = ScrollCursor { alias: remote.alias.clone(), scroll_id: None, buffer: Default::default(), exhausted: false };
        absorb_page(&mut c, page);
        st.clusters.push(c);
    }
    if st.total.is_null() {
        st.total = if params.get("rest_total_hits_as_int").is_some_and(|v| v == "true" || v.is_empty()) { json!(0) } else { json!({"value": 0, "relation": "eq"}) };
    }
    let hits = match next_page(node, req, &mut st).await {
        Ok(h) => h,
        Err(e) => return err(e),
    };
    let key = format!("{SCROLL_PREFIX}{}", elasticsearch_engine::index::new_uuid());
    let id = wrap_id(node, &key);
    let mut out = scroll_response(&id, started.elapsed().as_millis() as u64, st.shards, &st.total, hits);
    // `_clusters` after `_shards`.
    if let Some(o) = out.as_object_mut() {
        let hits = o.remove("hits");
        o.insert("_clusters".into(), clusters_json(&outcomes));
        if let Some(h) = hits {
            o.insert("hits".into(), h);
        }
    }
    scrolls().lock().insert(key, Arc::new(tokio::sync::Mutex::new(st)));
    node.engine.ccs_stats.lock().record(started.elapsed().as_millis() as u64, &aliases, &skipped, true, false, wildcard);
    crate::rest::util::postprocess(req, RestResp::json(200, &out))
}

fn purge_expired() {
    let now = std::time::Instant::now();
    scrolls().lock().retain(|_, s| s.try_lock().map_or(true, |s| s.expires > now));
}

/// `_search/scroll` with a cross-cluster scroll id (on its coordinating node).
pub(crate) async fn continue_scroll(node: &Arc<ClusterNode>, req: &RestReq, wrapped: &str, inner: &str, scroll: Option<String>) -> RestResp {
    purge_expired();
    let started = std::time::Instant::now();
    let Some(entry) = scrolls().lock().get(inner).cloned() else { return context_missing(wrapped) };
    let mut st = entry.lock().await;
    // Another user's scroll is a missing one, and is freed (ES frees a
    // scroll context on a failed access).
    let caller = front_caller(node, req).await;
    if node.engine.security.enabled() && caller_key(&caller) != st.owner {
        drop(st);
        let _ = free_scroll(node, req, inner).await;
        return context_missing(wrapped);
    }
    if let Some(k) = scroll {
        st.expires = std::time::Instant::now() + keep_alive_of(&k);
        st.keep_alive = k;
    } else {
        st.expires = std::time::Instant::now() + keep_alive_of(&st.keep_alive);
    }
    let hits = match next_page(node, req, &mut st).await {
        Ok(h) => h,
        Err(e) => return err(e),
    };
    let out = scroll_response(wrapped, started.elapsed().as_millis() as u64, st.shards, &st.total, hits);
    crate::rest::util::postprocess(req, RestResp::json(200, &out))
}

/// Free a cross-cluster scroll and every cluster's own scroll; the number
/// of contexts freed.
/// `DELETE _search/scroll` of a cross-cluster scroll (on its coordinating
/// node): only its owner frees it; another subject's scroll counts as
/// missing (nothing freed, ES's 404), as for engine scrolls.
pub(crate) async fn clear_scroll(node: &Arc<ClusterNode>, req: &RestReq, inner: &str) -> u64 {
    if node.engine.security.enabled() {
        let Some(entry) = scrolls().lock().get(inner).cloned() else { return 0 };
        let owner = entry.lock().await.owner.clone();
        let caller = front_caller(node, req).await;
        if caller_key(&caller) != owner {
            return 0;
        }
    }
    free_scroll(node, req, inner).await
}

pub(crate) async fn free_scroll(node: &Arc<ClusterNode>, req: &RestReq, inner: &str) -> u64 {
    let Some(entry) = scrolls().lock().remove(inner) else { return 0 };
    let st = entry.lock().await;
    let mut freed = 0u64;
    for c in &st.clusters {
        let Some(id) = &c.scroll_id else { continue };
        let body = json!({"scroll_id": [id]});
        if c.alias.is_empty() {
            let mut sub = req.clone();
            if sub.set_method("DELETE").and_then(|_| sub.set_uri("/_search/scroll")).and_then(|_| sub.set_json(&body)).is_err() {
                continue;
            }
            let r = Box::pin(context(node, &sub, ContextKind::ClearScroll)).await;
            freed += r.body_json().and_then(|v| v["num_freed"].as_u64()).unwrap_or(0);
        } else if let Some(svc) = rc::service(node) {
            let rr = RemoteRestReq { method: "DELETE".into(), uri: "/_search/scroll".into(), body: serde_json::to_vec(&body).unwrap_or_default(), identity: st.identity.clone(), parts: false, ..Default::default() };
            if let Ok(r) = svc.rest(&c.alias, rr, false).await {
                freed += r.json().and_then(|v| v["num_freed"].as_u64()).unwrap_or(0);
            }
        }
    }
    freed.max(1)
}

// --------------------------------------------------------- point in time

/// `POST /{expr}/_pit` naming remote clusters: a point in time on every
/// cluster, under one id.
async fn open_pit(node: &Arc<ClusterNode>, req: &RestReq, local: Vec<String>, remotes: Vec<ClusterPart>) -> RestResp {
    let identity = identity(node, req).await;
    let query = req.query().to_string();
    let mut ids = Map::new();
    let mut shards = [0u64; 4];
    let mut add = |v: &Value| {
        for (i, k) in ["total", "successful", "skipped", "failed"].iter().enumerate() {
            shards[i] += v["_shards"][k].as_u64().unwrap_or(0);
        }
    };
    if !local.is_empty() {
        let expr = local.join(",");
        let mut sub = req.clone();
        if let Err(e) = sub.set_uri(&with_query(format!("/{}/_pit", encode_targets(&expr)), &query)) {
            return err(e);
        }
        let r = Box::pin(super::read(node, &sub, &expr)).await;
        if !r.is_success() {
            return r;
        }
        let v = r.body_json().unwrap_or(Value::Null);
        add(&v);
        ids.insert(String::new(), v["id"].clone());
    }
    let results = futures::future::join_all(remotes.iter().map(|remote| {
        let uri = with_query(format!("/{}/_pit", encode_targets(&remote.indices.join(","))), &query);
        remote_call(node, remote, "POST", uri, req.body.to_vec(), identity.clone(), false)
    }))
    .await;
    for (remote, (r, _)) in remotes.iter().zip(results) {
        match remote_failure(&r) {
            Some(_) if remote.skip_unavailable => {}
            Some(e) => return err(e),
            None => {
                let v = r.ok().and_then(|r| r.json()).unwrap_or(Value::Null);
                add(&v);
                ids.insert(remote.alias.clone(), v["id"].clone());
            }
        }
    }
    let inner = format!("{PIT_PREFIX}{}", hex_encode(Value::Object(ids).to_string().as_bytes()));
    let id = wrap_id(node, &inner);
    RestResp::json(200, &json!({"id": id, "_shards": {"total": shards[0], "successful": shards[1], "skipped": shards[2], "failed": shards[3]}}))
}

use eremote::hex_encode;

/// The per-cluster ids of a cross-cluster point in time.
fn pit_ids(inner: &str) -> Option<Map<String, Value>> {
    eremote::ccs_pit_ids(inner)
}

/// A search on a cross-cluster point in time: every cluster searches its
/// own point in time with the shard body, and the shard results merge as a
/// cross-cluster search's (aggregations included).
pub(crate) async fn search_pit(node: &Arc<ClusterNode>, req: &RestReq, wrapped: &str, inner: &str) -> RestResp {
    let started = std::time::Instant::now();
    let Some(ids) = pit_ids(inner) else { return context_missing(wrapped) };
    let local: Vec<String> = if ids.contains_key("") { vec![String::new()] } else { Vec::new() };
    let remotes: Vec<ClusterPart> = ids.keys().filter(|a| !a.is_empty()).map(|alias| ClusterPart {
        alias: alias.clone(), indices: Vec::new(), skip_unavailable: eremote::remote_config(&node.engine, alias).skip_unavailable,
    }).collect();
    Box::pin(search_on(node, req, local, remotes, started, false, false, Some((wrapped.to_string(), ids)))).await
}

/// The shard result of a point in time held by one shard context, for a
/// coordinating cluster's cross-cluster reduce (its shard state kept).
pub(crate) fn single_pit_part(resp: RestResp) -> RestResp {
    if !resp.is_success() {
        return resp;
    }
    let mut sources = elasticsearch_engine::source_json::ResponseSources::default();
    let memory = elasticsearch_engine::memory::RequestMemory::new();
    let mut part = match parse_search_part(&mut sources, &resp.body, &memory) {
        Ok(v) => v,
        Err(e) => return err(e),
    };
    if let Some(o) = part.as_object_mut() {
        o.remove("pit_id");
    }
    let out = json!({"_ccs_parts": [part], "_ccs_failures": {"total": 0, "successful": 0, "skipped": 0, "failed": 0}, "took": 0});
    let mut raw = RestResp::json(200, &Value::Null);
    raw.replace_body_vec(sources.render(&out));
    raw
}

/// `DELETE /_pit` of a cross-cluster point in time.
pub(crate) async fn close_pit(node: &Arc<ClusterNode>, req: &RestReq, wrapped: &str, inner: &str) -> RestResp {
    let Some(ids) = pit_ids(inner) else {
        return RestResp::json(404, &json!({"succeeded": true, "num_freed": 0}));
    };
    let _ = wrapped;
    let identity = identity(node, req).await;
    let mut freed = 0u64;
    for (alias, id) in &ids {
        let body = json!({"id": id});
        if alias.is_empty() {
            let mut sub = req.clone();
            if sub.set_json(&body).is_err() {
                continue;
            }
            let r = Box::pin(context(node, &sub, ContextKind::ClosePit)).await;
            freed += r.body_json().and_then(|v| v["num_freed"].as_u64()).unwrap_or(0);
        } else if let Some(svc) = rc::service(node) {
            let rr = RemoteRestReq { method: "DELETE".into(), uri: "/_pit".into(), body: serde_json::to_vec(&body).unwrap_or_default(), identity: identity.clone(), parts: false, ..Default::default() };
            if let Ok(r) = svc.rest(alias, rr, false).await {
                freed += r.json().and_then(|v| v["num_freed"].as_u64()).unwrap_or(0);
            }
        }
    }
    RestResp::json(if freed == 0 { 404 } else { 200 }, &json!({"succeeded": true, "num_freed": freed}))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cluster-qualified names follow `is_remote_part`: a selector or date
    /// math holding `:` is a local name, never another cluster's.
    #[test]
    fn own_names_follow_the_remote_part_rules() {
        assert_eq!(own_name("b:docs", "b").as_deref(), Some("docs"));
        assert_eq!(own_name("b:docs", ""), None);
        assert_eq!(own_name("docs", "b"), None);
        assert_eq!(own_name("docs::failures", "").as_deref(), Some("docs::failures"));
        assert_eq!(own_name("docs::failures", "docs"), None);
        let math = "<logs-{now/d{yyyy.MM.dd|+12:00}}>";
        assert_eq!(own_name(math, "").as_deref(), Some(math));
        let body = serde_json::to_vec(&json!({"query": {"bool": {"should": [{"term": {"_index": "docs::failures"}}, {"terms": {"_index": ["b:docs", "docs"]}}]}}})).unwrap();
        let local: Value = serde_json::from_slice(&index_terms_for(&body, "")).unwrap();
        assert_eq!(local["query"]["bool"]["should"][0]["term"]["_index"], "docs::failures");
        assert_eq!(local["query"]["bool"]["should"][1]["terms"]["_index"][1], "docs");
        assert_ne!(local["query"]["bool"]["should"][1]["terms"]["_index"][0], "docs");
        let remote: Value = serde_json::from_slice(&index_terms_for(&body, "b")).unwrap();
        assert_eq!(remote["query"]["bool"]["should"][1]["terms"]["_index"][0], "docs");
    }

    #[test]
    fn scroll_owner_key_separates_api_keys_and_realms() {
        use elasticsearch_engine::security::{Authn, Principal};
        let user = |name: &str, realm: &str, ty: &str| Authn {
            username: name.into(), roles: vec![], full_name: None, email: None, metadata: json!({}), enabled: true,
            realm_name: realm.into(), realm_type: ty.into(), auth_type: "realm", principal: Principal::User, run_as: None,
        };
        let key = |id: &str| {
            let mut a = user("alice", "_es_api_key", "_es_api_key");
            a.auth_type = "api_key";
            a.principal = Principal::ApiKey { id: id.into(), name: "k".into(), metadata: json!({}) };
            a
        };
        assert_ne!(caller_key(&Some(key("k1"))), caller_key(&Some(key("k2"))), "two API keys of one owner shared a scroll owner");
        assert_eq!(caller_key(&Some(key("k1"))), caller_key(&Some(key("k1"))));
        assert_ne!(caller_key(&Some(user("alice", "file1", "file"))), caller_key(&Some(user("alice", "ldap1", "ldap"))));
        let mut ran = user("admin", "file1", "file");
        ran.run_as = Some(Box::new(user("alice", "native1", "native")));
        assert_ne!(caller_key(&Some(ran)), caller_key(&Some(user("alice", "native1", "native"))), "a run-as subject shared the target's scroll owner");
    }

    #[test]
    fn clusters_section_counts_and_orders_as_es() {
        let outcomes = vec![
            Outcome { alias: String::new(), indices: "logs".into(), status: "successful", took: Some(3), timed_out: false, shards: Some([2, 2, 0, 0]), failures: vec![] },
            Outcome { alias: "b".into(), indices: "logs*".into(), status: "partial", took: Some(5), timed_out: false, shards: Some([3, 2, 0, 1]), failures: vec![json!({"shard": 1})] },
            Outcome { alias: "c".into(), indices: "x".into(), status: "skipped", took: None, timed_out: false, shards: None, failures: vec![cluster_failure(&EsError::new(502, "connect_transport_exception", "Unable to connect to [c]"))] },
        ];
        let v = clusters_json(&outcomes);
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["total", "successful", "skipped", "running", "partial", "failed", "details"]);
        assert_eq!((v["total"].clone(), v["successful"].clone(), v["skipped"].clone(), v["partial"].clone(), v["failed"].clone()), (json!(3), json!(1), json!(1), json!(1), json!(0)));
        assert_eq!(v["details"]["(local)"]["_shards"]["total"], 2);
        assert_eq!(v["details"]["b"]["failures"][0]["shard"], 1);
        assert_eq!(v["details"]["c"]["failures"][0]["reason"]["type"], "connect_transport_exception");
        assert!(v["details"]["c"].get("took").is_none());
        let d: Vec<&str> = v["details"]["b"].as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(d, vec!["status", "indices", "took", "timed_out", "_shards", "failures"]);
    }

    #[test]
    fn remote_parts_name_their_cluster() {
        let mut part = json!({"hits": {"hits": [{"_index": "logs", "_id": "1", "inner_hits": {"x": {"hits": {"hits": [{"_index": "logs"}]}}}}]},
            "_shards": {"failures": [{"index": "logs", "shard": 0}]}});
        label_remote(&mut part, "b");
        assert_eq!(part["hits"]["hits"][0]["_index"], "b:logs");
        assert_eq!(part["hits"]["hits"][0]["inner_hits"]["x"]["hits"]["hits"][0]["_index"], "b:logs");
        assert_eq!(part["_shards"]["failures"][0]["index"], "b:logs");
        // Already labelled stays.
        label_remote(&mut part, "b");
        assert_eq!(part["hits"]["hits"][0]["_index"], "b:logs");
    }

    #[test]
    fn cross_cluster_pit_ids_round_trip() {
        let ids = json!({"": "n.~pabc", "b": "remote-pit"});
        let inner = format!("{PIT_PREFIX}{}", hex_encode(ids.to_string().as_bytes()));
        assert_eq!(pit_ids(&inner).unwrap()["b"], "remote-pit");
        assert!(!eremote::remote_only_pit(&format!("AAAAAAAAAAAAAAAAAAAAAA.{inner}")));
        let remote = format!("{PIT_PREFIX}{}", hex_encode(json!({"b": "r"}).to_string().as_bytes()));
        assert!(eremote::remote_only_pit(&format!("AAAAAAAAAAAAAAAAAAAAAA.{remote}")));
        assert!(!eremote::remote_only_pit("AAAAAAAAAAAAAAAAAAAAAA.~p123"));
    }
}

// ------------------------------------------------------------------ ES|QL

/// A remote cluster's shard reads of an ES|QL query (`FROM alias:index`).
pub const ACTION_ESQL_FETCH: &str = "indices:data/read/esql/cluster_fetch";

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct EsqlFetchReq {
    pub identity: Option<String>,
    /// The `FetchSpec` (JSON), its shards named as the remote cluster names them.
    pub spec: String,
}

/// The identity of an ES|QL query's caller (its `_query` request).
async fn template_identity(node: &ClusterNode, template: &RestReq) -> Option<String> {
    identity(node, template).await
}

/// `FROM alias:index`: the remote cluster's indices as schema-only targets
/// (`alias:index`), from its mappings and settings read as the caller. An
/// unavailable `skip_unavailable` remote contributes nothing.
pub(crate) async fn esql_remote_targets(node: &Arc<ClusterNode>, template: &RestReq, alias: &str, indices: &[String]) -> Result<Vec<elasticsearch_engine::search::Target>, EsError> {
    let skip = eremote::remote_config(&node.engine, alias).skip_unavailable;
    let svc = rc::service(node).ok_or_else(|| eremote::connection_error(&node.engine, alias))?;
    let identity = template_identity(node, template).await;
    let expr = encode_targets(&indices.join(","));
    let get = |uri: String| RemoteRestReq { method: "GET".into(), uri, body: Vec::new(), identity: identity.clone(), ..Default::default() };
    let mappings = svc.rest(alias, get(format!("/{expr}/_mapping?expand_wildcards=open")), !skip).await;
    let settings = svc.rest(alias, get(format!("/{expr}/_settings?flat_settings=true&expand_wildcards=open")), !skip).await;
    let (m, st) = match (remote_failure(&mappings), remote_failure(&settings)) {
        (None, None) => (mappings.ok().and_then(|r| r.json()).unwrap_or(Value::Null), settings.ok().and_then(|r| r.json()).unwrap_or(Value::Null)),
        (Some(e), _) | (_, Some(e)) => {
            if skip {
                tracing::debug!(%alias, "ES|QL: remote cluster skipped: {e}");
                return Ok(Vec::new());
            }
            return Err(e);
        }
    };
    let mut out = Vec::new();
    for (index, mapping) in m.as_object().into_iter().flatten() {
        let settings = st[index]["settings"].clone();
        let meta = json!({"settings": settings, "mappings": mapping["mappings"]});
        out.push(elasticsearch_engine::esql::remote::remote_shell(&format!("{alias}:{index}"), &meta)?);
    }
    Ok(out)
}

/// Read remote clusters' shards of an ES|QL query: each cluster serves its
/// part of `spec` across its own nodes ([`ACTION_ESQL_FETCH`]).
pub(crate) async fn esql_remote_fetch(node: &Arc<ClusterNode>, template: &RestReq, spec: &elasticsearch_engine::esql::remote::FetchSpec) -> Result<Vec<elasticsearch_engine::esql::remote::ShardResult>, EsError> {
    use elasticsearch_engine::esql::remote::{FetchSpec, RemoteShard, ShardResult};
    let identity = template_identity(node, template).await;
    let svc = rc::service(node).ok_or_else(|| EsError::new(502, "connect_transport_exception", "no remote cluster service"))?;
    let mut by_alias: BTreeMap<String, Vec<RemoteShard>> = BTreeMap::new();
    for s in &spec.shards {
        let (alias, index) = s.index.split_once(':').unwrap_or(("", &s.index));
        by_alias.entry(alias.to_string()).or_default().push(RemoteShard { index: index.to_string(), ..s.clone() });
    }
    let mut out = Vec::new();
    for (alias, shards) in by_alias {
        let part = FetchSpec { shards, ..spec.clone() };
        let req = EsqlFetchReq { identity: rc::outgoing_identity(node, &alias, identity.clone()), spec: serde_json::to_string(&part).map_err(|e| EsError::internal(e.to_string()))? };
        let skip = eremote::remote_config(&node.engine, &alias).skip_unavailable;
        let answer: Result<String, EsError> = svc.send(&alias, ACTION_ESQL_FETCH, &req, Some(Duration::from_secs(300)), !skip).await;
        let text = answer?;
        let mut results: Vec<ShardResult> = serde_json::from_str(&text).map_err(|e| EsError::internal(format!("bad remote ES|QL shard read reply: {e}")))?;
        for r in results.iter_mut() {
            r.index = format!("{alias}:{}", r.index);
            for h in r.hits.iter_mut() {
                if let Some(i) = h.get("_index").and_then(Value::as_str).map(|i| format!("{alias}:{i}")) {
                    h["_index"] = json!(i);
                }
            }
        }
        out.extend(results);
    }
    Ok(out)
}

type PitNodes = parking_lot::Mutex<std::collections::HashMap<String, String>>;

/// The node each remote ES|QL fold point in time was opened on (they span
/// several fetch requests).
fn remote_pit_nodes() -> &'static PitNodes {
    static P: std::sync::OnceLock<PitNodes> = std::sync::OnceLock::new();
    P.get_or_init(Default::default)
}

/// Serve [`ACTION_ESQL_FETCH`] (registered for remote clusters).
pub(crate) fn register_esql(node: &Arc<ClusterNode>) {
    let n = node.clone();
    node.transport.register_handler(ACTION_ESQL_FETCH, move |req: EsqlFetchReq| {
        let n = n.clone();
        let peer = elasticsearch_transport::inbound_remote_peer();
        async move {
            let spec: elasticsearch_engine::esql::remote::FetchSpec = serde_json::from_str(&req.spec)
                .map_err(|e| elasticsearch_transport::RemoteError::new("illegal_argument_exception", format!("bad ES|QL fetch: {e}"), 400))?;
            let mut headers: Vec<(String, String)> = vec![("content-type".into(), "application/json".into())];
            if n.engine.security.enabled() {
                let stored = rc::authenticate_remote(&n, &peer, &req.identity).map_err(|e| rc::to_remote(&e))?
                    .ok_or_else(|| elasticsearch_transport::RemoteError::new("security_exception", "missing authentication credentials for a cross-cluster ES|QL read", 401))?;
                let authn = stored.to_authn();
                let perm = n.engine.security.permission(&authn);
                let denied: Vec<String> = spec.shards.iter().map(|s| s.index.clone()).filter(|i| !perm.check_index("indices:data/read/esql", i)).collect();
                if !denied.is_empty() {
                    return Err(elasticsearch_transport::RemoteError::new("security_exception", format!(
                        "action [indices:data/read/esql] is unauthorized for user [{}] with effective roles [{}] on indices [{}]",
                        authn.username, perm.effective_roles.join(","), denied.join(",")), 403));
                }
                headers.push((super::IDENTITY_HEADER.into(), stored.encode()));
            }
            // The shard reads are `POST /_query` fetches (their spec in a header);
            // the request still carries a body, as every `_query` does.
            let template = RestReq::try_new("POST", "/_query", &headers, br#"{"query":"ROW 1"}"#, 1, Default::default(), None).map_err(|e| rc::to_remote(&e))?;
            let pit_nodes: std::collections::HashMap<String, String> = remote_pit_nodes().lock().clone();
            let source = super::esql_dist::EsqlShards { node: n.clone(), state: n.state(), engines: n.all_shard_engines(), template,
                runtime: tokio::runtime::Handle::current(), seen: Default::default(), may_forward: false, pit_nodes: std::sync::Mutex::new(pit_nodes) };
            let results = tokio::task::spawn_blocking(move || {
                let mut memory = elasticsearch_engine::memory::breakers().reserve_request(0, "remote ES|QL shard reads")?;
                let r = source.fetch_cluster(&spec, &mut memory);
                if let Ok(nodes) = source.pit_nodes.lock() {
                    let mut all = remote_pit_nodes().lock();
                    for (k, v) in nodes.iter() {
                        all.insert(k.clone(), v.clone());
                    }
                    if let Some(p) = spec.pit.as_ref().filter(|p| p.phase == elasticsearch_engine::esql::remote::PitPhase::Close) {
                        all.remove(&p.id);
                    }
                }
                r
            }).await.map_err(|e| elasticsearch_transport::RemoteError::internal(e.to_string()))?.map_err(|e| rc::to_remote(&e))?;
            serde_json::to_string(&results).map_err(|e| elasticsearch_transport::RemoteError::internal(e.to_string()))
        }
    });
    node.transport.allow_remote_action(ACTION_ESQL_FETCH);
}
