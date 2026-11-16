//! `_shrink` / `_split` / `_clone` under the cluster.
//!
//! The engine implements a resize as "create the target, then copy the
//! source's documents into it", all inside one engine. As a metadata
//! request it executes on the elected master's engine, so the copied
//! documents exist only there, and only if the master happens to hold a
//! copy of the source. The published target index then gets a fresh primary
//! on whichever node the allocator picks: unless that is the master, the
//! primary starts from an empty shell (and a replica on the master is wiped
//! by peer recovery), so the target had no documents.
//!
//! Here the resize still creates the target through the master, and then
//! the documents are copied through the ordinary cluster paths: a scroll
//! over the source (served by a node holding a source copy) and bulk writes
//! to the target primary, which replicate like any other write. Documents
//! keep their id, routing and version (`version_type: external`), which
//! also makes the copy a no-op for documents the target primary already has
//! (the primary landed on the master).
//!
//! The engine's resize copies each source shard dir into the target shards
//! it splits into and then deletes, per target shard, the documents that
//! route elsewhere (ES's split delete-by-routing). Engine 0 holds every
//! document of cluster source shard 0 in its local shard 0, so for a split
//! the master's engine 0 ends with target shard 0 clean but with local
//! target shards `1..T/S` holding those documents too. A whole-index search
//! of engine 0 (cluster shard 0) covers every local shard, so they would
//! show up as duplicates: [`purge_misplaced`] removes them through engine 0
//! by id before the resize answers. The copy itself writes them to their
//! own cluster shard.

//!
//! A resize answers success only once the target is complete: every
//! target primary started, every source document copied (no pipeline runs
//! on the copies: ES resizes shard files), misplaced copies removed, and
//! the target verified to hold exactly the source's documents — same ids,
//! routing, versions and sources, each on the shard its routing selects.
//! Then it waits for `wait_for_active_shards` like ES's create-index
//! response (`shards_acknowledged`). The copy is tracked in the cluster
//! state (`resize` progress, [`PROGRESS_KIND`]) while it runs. If any step
//! fails (a primary that does not start within `timeout`, a rejected bulk
//! item, a lost node, a mismatch), or the master that ran it is gone, the
//! resize fails and the target is left explicitly incomplete: read and
//! write blocked, its progress entry `failed` with the reason, to be
//! deleted by the user. Success is never reported for a partial copy.

use crate::front::{dispatch, on_master_route, RestReq, RestResp};
use crate::node::ClusterNode;
use crate::snapshot_api::common::{self as progress, PROGRESS_CUSTOM};
use axum::http::Method;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// The resize progress entries' kind in the progress custom.
pub const PROGRESS_KIND: &str = "resize";

/// Test seam (hidden): the next `n` copied documents are sent with a
/// source the target primary rejects (a metadata field inside the
/// document), so their bulk items fail while the others succeed.
#[doc(hidden)]
pub static REJECT_COPY_ITEMS: AtomicU32 = AtomicU32::new(0);

/// Resizes this process runs, by (master node id, target index uuid).
fn running() -> &'static parking_lot::Mutex<HashSet<(String, String)>> {
    static RUNNING: std::sync::OnceLock<parking_lot::Mutex<HashSet<(String, String)>>> = std::sync::OnceLock::new();
    RUNNING.get_or_init(Default::default)
}

const PAGE: usize = 500;

/// `(source, target)` when `req` is `PUT|POST /{source}/_shrink|_split|_clone/{target}`.
pub fn resize_target(req: &RestReq) -> Option<(String, String)> {
    let m = req.method();
    if m != Method::PUT && m != Method::POST {
        return None;
    }
    let segs = req.segments();
    if segs.len() != 3 || segs[0].starts_with('_') || !matches!(segs[1].as_str(), "_shrink" | "_split" | "_clone") {
        return None;
    }
    Some((segs[0].clone(), segs[2].clone()))
}

tokio::task_local! {
    /// The resize caller's credential headers: its sub-requests (settings,
    /// copy, cleanup) are authorized as the caller.
    static CREDENTIALS: crate::front::SharedHeaders;
}

fn internal(method: &str, uri: std::fmt::Arguments<'_>, body: Vec<u8>, ndjson: bool) -> elasticsearch_engine::error::Result<RestReq> {
    let ct = if ndjson { "application/x-ndjson" } else { "application/json" };
    let request=CREDENTIALS.try_with(|headers|RestReq::try_new_uri(method,uri,headers,&body,0,Default::default(),None))
        .unwrap_or_else(|_|RestReq::try_new_uri(method,uri,&[],&body,0,Default::default(),None));
    request.and_then(|mut req|{req.push_header("content-type",ct)?;Ok(req)})
}

/// `dispatch` behind a boxed future: `dispatch` calls [`handle`], which
/// calls back into `dispatch` for the copy.
fn send(node: &Arc<ClusterNode>, req: elasticsearch_engine::error::Result<RestReq>) -> std::pin::Pin<Box<dyn std::future::Future<Output = RestResp> + Send>> {
    let node = node.clone();
    Box::pin(async move { match req { Ok(req) => dispatch(&node, req).await, Err(e) => RestResp::json(e.status, &e.to_json()) } })
}

fn call(node: &Arc<ClusterNode>, method: &str, uri: std::fmt::Arguments<'_>, body: Option<Value>) -> std::pin::Pin<Box<dyn std::future::Future<Output=RestResp>+Send>> {
    let request=internal(method,uri,Vec::new(),false).and_then(|mut req| {if let Some(body)=body {req.set_json(&body)?;} Ok(req)});
    send(node,request)
}

async fn wait_for_primary(node: &ClusterNode, index: &str, uuid: &str, timeout: Duration) -> Result<(), String> {
    let mut rx = node.subscribe();
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        // Every target shard: the copy bulk-routes documents across all of
        // them (a split / shrink target has its own shard count).
        let ok = {
            let st = rx.borrow();
            if !st.metadata.indices.get(index).is_some_and(|m| m.uuid == uuid) {
                return Err(format!("the target index [{index}] was deleted"));
            }
            let n = crate::shard_engines::shards_of(&st.metadata, index).max(1);
            (0..n).all(|k| st.routing_table.shard_table(index, k).and_then(|t| t.active_primary()).is_some())
        };
        if ok {
            return Ok(());
        }
        if tokio::time::timeout_at(deadline, rx.changed()).await.is_err() {
            return Err(format!("the target's primaries did not all start within [{}ms]", timeout.as_millis()));
        }
    }
}

/// ES `ActiveShardCount` of `wait_for_active_shards` (default 1): the
/// active copies every shard needs (`None`: `all`).
fn active_shard_count(v: Option<&String>) -> Result<Option<u32>, String> {
    match v.map(String::as_str) {
        None | Some("") => Ok(Some(1)),
        Some("all") => Ok(None),
        Some(n) => n.parse::<u32>().map(Some).map_err(|_| format!("failed to parse value [{n}] for [wait_for_active_shards]")),
    }
}

/// Wait (up to `timeout`) until every shard of `index` has `count` active
/// copies (`None`: all of them).
async fn wait_for_active(node: &ClusterNode, index: &str, count: Option<u32>, timeout: Duration) -> bool {
    let mut rx = node.subscribe();
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let ok = {
            let st = rx.borrow();
            let Some(im) = st.metadata.indices.get(index) else { return false };
            let need = count.unwrap_or(im.number_of_replicas + 1) as usize;
            (0..im.number_of_shards.max(1)).all(|k| st.routing_table.shard_table(index, k).is_some_and(|t| t.active_copies().count() >= need))
        };
        if ok {
            return true;
        }
        if tokio::time::timeout_at(deadline, rx.changed()).await.is_err() {
            return false;
        }
    }
}

pub async fn handle(node: &Arc<ClusterNode>, req: &RestReq, source: &str, target: &str) -> RestResp {
    match crate::reindex_multi::credentials(req) {Ok(creds)=>CREDENTIALS.scope(creds,handle_as_caller(node, req, source, target)).await,Err(e)=>RestResp::json(e.status,&e.to_json())}
}

async fn handle_as_caller(node: &Arc<ClusterNode>, req: &RestReq, source: &str, target: &str) -> RestResp {
    let params = req.params();
    let active = match active_shard_count(params.get("wait_for_active_shards")) {
        Ok(a) => a,
        Err(e) => return RestResp::error(400, "illegal_argument_exception", &e),
    };
    let timeout = params.get("timeout").and_then(|t| elasticsearch_engine::date::parse_time_value(t)).map(|ms| Duration::from_millis(ms.max(0) as u64)).unwrap_or(Duration::from_secs(30));
    // Sparse opens: the master's engine 0 may hold the source without its
    // shard copies open; the engine's resize needs every source shard.
    let shells = if node.is_master() { node.open_resize_sources(source) } else { Vec::new() };
    let resp = on_master_route(node, req).await;
    node.close_resize_sources(source, &shells);
    // A forwarded resize was executed, and its documents copied, by the
    // master (which re-enters here through `dispatch`).
    if !resp.is_success() || !node.is_master() {
        return resp;
    }
    let Some(uuid) = node.state().metadata.indices.get(target).map(|m| m.uuid.clone()) else {
        return RestResp::error(500, "illegal_state_exception", &format!("resize of [{source}] to [{target}] did not complete: the target index was deleted"));
    };
    let key = (node.local.id.clone(), uuid.clone());
    running().lock().insert(key.clone());
    let done = complete(node, source, target, &uuid, timeout).await;
    let resp = match done {
        Ok(()) => {
            let mut body = resp.body_json().unwrap_or_else(|| json!({"acknowledged": true, "index": target}));
            // ES: the response waits for the active shards; a timeout is
            // `shards_acknowledged: false` (the copy itself is complete).
            body["shards_acknowledged"] = json!(wait_for_active(node, target, active, timeout).await);
            RestResp::json(200, &body)
        }
        Err(reason) => {
            tracing::warn!(source, target, "resize did not complete; the target is left incomplete: {reason}");
            let marked = mark_incomplete(node, target, &uuid, &reason).await;
            let left = match marked {
                Ok(()) => format!("the target index [{target}] is incomplete: it was left read and write blocked and must be deleted"),
                Err(e) => format!("the target index [{target}] is incomplete and could not be blocked ({e}): delete it"),
            };
            RestResp::error(500, "illegal_state_exception", &format!("resize of [{source}] to [{target}] did not complete: {reason}; {left}"))
        }
    };
    running().lock().remove(&key);
    resp
}

/// Make target incarnation `uuid` hold exactly the documents of `source`.
async fn complete(node: &Arc<ClusterNode>, source: &str, target: &str, uuid: &str, timeout: Duration) -> Result<(), String> {
    let entry = json!({"target": target, "index_uuid": uuid, "source": source, "master": node.local.id, "state": "copying"});
    progress::update_custom(node, PROGRESS_CUSTOM, move |v| {
        if !v.is_object() { *v = json!({}); }
        if !v[PROGRESS_KIND].is_array() { v[PROGRESS_KIND] = json!([]); }
        let a = v[PROGRESS_KIND].as_array_mut().expect("array");
        a.retain(|e| e["index_uuid"] != entry["index_uuid"]);
        a.push(entry);
        Ok(true)
    }).await.map_err(|e| format!("cannot record the resize: {e}"))?;
    wait_for_primary(node, target, uuid, timeout).await?;
    // The target inherits the source's write block (a resize source must be
    // read-only); lift it for the copy and put it back afterwards.
    let prepared_call = call(node, "GET", format_args!("/{target}/_settings?flat_settings=true"), None);
    let settings = prepared_call.await;
    if !settings.is_success() {
        return Err(format!("reading the target's settings answered {}", settings.status));
    }
    let blocked = settings.body_json().map(|v| {
        let b = &v[target]["settings"]["index.blocks.write"];
        b == &json!("true") || b == &json!(true)
    }).unwrap_or(false);
    if blocked {
        let prepared_call = call(node, "PUT", format_args!("/{target}/_settings"), Some(json!({"index.blocks.write": false})));
        let r = prepared_call.await;
        if !r.is_success() {
            return Err(format!("lifting the target's write block answered {}", r.status));
        }
    }
    let (copied, misplaced, expected) = copy_docs(node, source, target).await?;
    tracing::debug!(source, target, copied, "resize: documents copied to the target primaries");
    purge_misplaced(node, target, &misplaced).await.map_err(|e| format!("removing documents of other shards from target shard 0 failed: {e}"))?;
    let found = target_digest(node, target).await?;
    if found != expected {
        return Err(format!("the target holds {} documents (digest {:032x}), the source {} (digest {:032x})", found.count, found.sum, expected.count, expected.sum));
    }
    if blocked {
        let prepared_call = call(node, "PUT", format_args!("/{target}/_settings"), Some(json!({"index.blocks.write": true})));
        let r = prepared_call.await;
        if !r.is_success() {
            return Err(format!("restoring the target's write block answered {}", r.status));
        }
    }
    let uuid = uuid.to_string();
    progress::update_custom(node, PROGRESS_CUSTOM, move |v| {
        let Some(a) = v.get_mut(PROGRESS_KIND).and_then(Value::as_array_mut) else { return Ok(false) };
        let before = a.len();
        a.retain(|e| e["index_uuid"].as_str() != Some(uuid.as_str()));
        let changed = a.len() != before;
        if a.is_empty() {
            if let Some(o) = v.as_object_mut() { o.remove(PROGRESS_KIND); }
        }
        Ok(changed)
    }).await.map_err(|e| format!("cannot record the completed resize: {e}"))
}

/// Leave target incarnation `uuid` explicitly incomplete: read and write
/// blocked, its progress entry `failed` with `reason`.
async fn mark_incomplete(node: &Arc<ClusterNode>, target: &str, uuid: &str, reason: &str) -> Result<(), String> {
    let (u, why) = (uuid.to_string(), reason.to_string());
    let recorded = progress::update_custom(node, PROGRESS_CUSTOM, move |v| {
        let Some(e) = v.get_mut(PROGRESS_KIND).and_then(Value::as_array_mut).and_then(|a| a.iter_mut().find(|e| e["index_uuid"].as_str() == Some(u.as_str()))) else { return Ok(false) };
        e["state"] = json!("failed");
        e["reason"] = json!(why);
        Ok(true)
    }).await;
    if !node.state().metadata.indices.get(target).is_some_and(|m| m.uuid == uuid) {
        return recorded;
    }
    let prepared_call = call(node, "PUT", format_args!("/{target}/_settings"), Some(json!({"index.blocks.write": true, "index.blocks.read": true})));
    let r = prepared_call.await;
    if !r.is_success() {
        return Err(format!("blocking the target answered {}", r.status));
    }
    recorded
}

/// Master only (the 1 s master tick): a resize whose master is gone left
/// its target with an unknown part of the documents: it is marked
/// incomplete. Entries of deleted targets are dropped.
pub async fn tick(node: &Arc<ClusterNode>) {
    let st = node.state();
    let entries = progress::progress_entries(&st, PROGRESS_KIND);
    if entries.is_empty() {
        return;
    }
    let mut gone = Vec::new();
    for e in entries {
        let (Some(target), Some(uuid)) = (e["target"].as_str(), e["index_uuid"].as_str()) else { continue };
        if !st.metadata.indices.get(target).is_some_and(|m| m.uuid == uuid) {
            gone.push(uuid.to_string());
            continue;
        }
        let ours = running().lock().contains(&(node.local.id.clone(), uuid.to_string()));
        if e["state"] == "copying" && !ours {
            tracing::warn!(target, "resize was orphaned by a master change; its target is left incomplete");
            if let Err(err) = mark_incomplete(node, target, uuid, "the master running the resize left before it completed").await {
                tracing::warn!(target, "cannot mark the orphaned resize target incomplete: {err}");
            }
        }
    }
    if !gone.is_empty() {
        let _ = progress::update_custom(node, PROGRESS_CUSTOM, move |v| {
            let Some(a) = v.get_mut(PROGRESS_KIND).and_then(Value::as_array_mut) else { return Ok(false) };
            let before = a.len();
            a.retain(|e| !e["index_uuid"].as_str().is_some_and(|u| gone.iter().any(|g| g == u)));
            let changed = a.len() != before;
            if a.is_empty() {
                if let Some(o) = v.as_object_mut() { o.remove(PROGRESS_KIND); }
            }
            Ok(changed)
        }).await;
    }
}

/// An order-independent digest of a set of documents: their count and
/// the sum of a 128-bit hash of each one's id, routing, version and source.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Digest {
    count: u64,
    sum: u128,
}

impl Digest {
    fn add(&mut self, hit: &Value) {
        let half = |salt: u8| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            salt.hash(&mut h);
            hit["_id"].as_str().hash(&mut h);
            hit["_routing"].as_str().hash(&mut h);
            hit["_version"].as_u64().hash(&mut h);
            hit["_source"].to_string().hash(&mut h);
            h.finish()
        };
        let v = (u128::from(half(1)) << 64) | u128::from(half(2));
        self.count += 1;
        self.sum = self.sum.wrapping_add(v);
    }
}

/// Every document of `target`, read shard by shard: each must be on the
/// shard its id and routing select.
async fn target_digest(node: &Arc<ClusterNode>, target: &str) -> Result<Digest, String> {
    let md = node.state().metadata.clone();
    let n = crate::shard_engines::shards_of(&md, target);
    let mut digest = Digest::default();
    for k in 0..n {
        let path = format!("/{target}/_search?scroll=1m&size={PAGE}&expand_wildcards=all&preference=_shards:{k}");
        scan(node, &path, |hits| {
            for h in hits {
                let id = h["_id"].as_str().unwrap_or_default();
                match crate::shard_engines::doc_shard(&md, target, id, h["_routing"].as_str()) {
                    Ok(s) if s == k => digest.add(h),
                    Ok(s) => return Err(format!("document [{id}] is on target shard [{k}] but routes to [{s}]")),
                    Err(e) => return Err(format!("document [{id}] does not route: {e:?}")),
                }
            }
            Ok(())
        }).await?;
    }
    Ok(digest)
}

/// A scroll over `path` (`_search?scroll=…`), a page at a time.
struct Scroll<'a> {
    node: &'a Arc<ClusterNode>,
    path: String,
    id: Option<String>,
    started: bool,
}

impl<'a> Scroll<'a> {
    fn new(node: &'a Arc<ClusterNode>, path: String) -> Scroll<'a> {
        Scroll { node, path, id: None, started: false }
    }

    /// The next page of hits (empty: the end). A failed shard is an error:
    /// its documents would be missing.
    async fn next(&mut self) -> Result<Vec<Value>, String> {
        // Finish building the request before suspending (fmt::Arguments is not Send).
        let prepared_call = if !self.started {
            self.started = true;
            // Vectors come back in `_source` (`exclude_vectors: false`): the
            // index default leaves them out, and both the copy and its
            // digest must hold them (production review #485).
            // And the indexing input (`_recovery_source`) where the mapping
            // filters `_source`: the copy indexes every field the source
            // did, as ES's file-level resize keeps them.
            {
                let request = internal("POST", format_args!("{}", self.path), Vec::new(), false).and_then(|mut req| {
                    req.set_json(&json!({"version": true, "sort": ["_doc"], "_source": {"exclude_vectors": false}}))?;
                    req.push_header(elasticsearch_http::RECOVERY_SOURCE_HEADER, "true")?;
                    Ok(req)
                });
                send(self.node, request)
            }
        } else {
            let Some(id) = self.id.clone() else { return Ok(Vec::new()) };
            call(self.node, "POST", format_args!("/_search/scroll"), Some(json!({"scroll": "1m", "scroll_id": id})))
        };
        let resp = prepared_call.await;
        if !resp.is_success() {
            return Err(format!("search [{}] answered {}", self.path, resp.status));
        }
        let mut page = resp.body_json().unwrap_or(Value::Null);
        if page["_shards"]["failed"].as_u64().unwrap_or(0) > 0 {
            return Err(format!("search [{}] failed on a shard: {}", self.path, page["_shards"]));
        }
        if let Some(id) = page["_scroll_id"].as_str() {
            self.id = Some(id.to_string());
        }
        Ok(match page["hits"]["hits"].take() { Value::Array(a) => a, _ => Vec::new() })
    }

    async fn close(self) {
        if let Some(id) = self.id {
            let prepared_call = call(self.node, "DELETE", format_args!("/_search/scroll"), Some(json!({"scroll_id": id})));
            let _ = prepared_call.await;
        }
    }
}

/// Scroll `path` to the end, `each` page of hits.
async fn scan(node: &Arc<ClusterNode>, path: &str, mut each: impl FnMut(&[Value]) -> Result<(), String>) -> Result<(), String> {
    let mut scroll = Scroll::new(node, path.to_string());
    let result = loop {
        match scroll.next().await {
            Ok(hits) if hits.is_empty() => break Ok(()),
            Ok(hits) => {
                if let Err(e) = each(&hits) {
                    break Err(e);
                }
            }
            Err(e) => break Err(e),
        }
    };
    scroll.close().await;
    result
}

/// Delete `ids` from engine 0's copy of `target` only (the shard header pins
/// the by-query to cluster shard 0's primary). The ids route to other target
/// shards, so the engine finds them in its local shards other than 0.
async fn purge_misplaced(node: &Arc<ClusterNode>, target: &str, ids: &[String]) -> Result<(), String> {
    for chunk in ids.chunks(1000) {
        let mut req = internal("POST", format_args!("/{target}/_delete_by_query?conflicts=proceed&refresh=true"), Vec::new(), false).and_then(|req|req.try_with_encoded_body(|out| {
            #[derive(serde::Serialize)] struct Ids<'a>{values:&'a [String]}
            #[derive(serde::Serialize)] struct Query<'a>{ids:Ids<'a>}
            #[derive(serde::Serialize)] struct Body<'a>{query:Query<'a>}
            serde_json::to_writer(out,&Body{query:Query{ids:Ids{values:chunk}}}).map_err(std::io::Error::other)
        })).map_err(|e| e.to_string())?;
        req.set_shard(0).map_err(|e| e.to_string())?;
        // Internal (hops > 0): `dispatch` keeps the shard header.
        req.hops = 1;
        let r = send(node, Ok(req)).await;
        if !r.is_success() {
            return Err(format!("delete_by_query on [{target}][0] answered {}", r.status));
        }
    }
    Ok(())
}

/// Copy every document of `source` that `target` does not have yet.
/// Also returns the ids that route to a target shard other than 0 (and
/// never to 0): the engine's split may have left them in engine 0's other
/// local target shards; and the digest of the source's documents.
async fn copy_docs(node: &Arc<ClusterNode>, source: &str, target: &str) -> Result<(u64, Vec<String>, Digest), String> {
    let md = node.state().metadata.clone();
    let multi = crate::shard_engines::shards_of(&md, target) > 1;
    let (mut elsewhere, mut in_zero) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
    // The source is write-blocked; make everything it holds searchable.
    let prepared_call = call(node, "POST", format_args!("/{source}/_refresh"), None);
    let refreshed = prepared_call.await;
    if !refreshed.is_success() {
        return Err(format!("refresh of [{source}] answered {}", refreshed.status));
    }
    let mut copied = 0u64;
    let mut digest = Digest::default();
    // The scroll reads a page, the bulk writes it: one page at a time.
    let mut scroll = Scroll::new(node, format!("/{source}/_search?scroll=1m&size={PAGE}&expand_wildcards=all"));
    let result = copy_pages(node, &mut scroll, &md, target, multi, &mut digest, &mut copied, &mut in_zero, &mut elsewhere).await;
    scroll.close().await;
    result?;
    let prepared_call = call(node, "POST", format_args!("/{target}/_refresh"), None);
    let refreshed = prepared_call.await;
    if !refreshed.is_success() {
        return Err(format!("refresh of [{target}] answered {}", refreshed.status));
    }
    // An id that also routes to shard 0 (under another routing) stays.
    Ok((copied, elsewhere.difference(&in_zero).cloned().collect(), digest))
}

#[allow(clippy::too_many_arguments)]
async fn copy_pages(node: &Arc<ClusterNode>, scroll: &mut Scroll<'_>, md: &elasticsearch_cluster::state::Metadata, target: &str, multi: bool, digest: &mut Digest, copied: &mut u64,
    in_zero: &mut std::collections::BTreeSet<String>, elsewhere: &mut std::collections::BTreeSet<String>) -> Result<(), String> {
    loop {
        let hits = scroll.next().await?;
        if hits.is_empty() {
            return Ok(());
        }
        let hits = &hits;
        for h in hits {
            let Some(id) = h["_id"].as_str() else { continue };
            digest.add(h);
            if multi {
                match crate::shard_engines::doc_shard(md, target, id, h["_routing"].as_str()) {
                    Ok(0) => {
                        in_zero.insert(id.to_string());
                    }
                    Ok(_) => {
                        elsewhere.insert(id.to_string());
                    }
                    Err(_) => {}
                }
            }
        }
        // No pipeline runs on the copies (the target inherits the source's
        // default / final pipelines; ES copies shard files).
        let routings: Vec<Option<String>> = hits.iter().filter(|h| h["_id"].is_string()).map(|h| h["_routing"].as_str().map(str::to_string)).collect();
        // Decided once: the body is encoded twice (sized, then written).
        let rejected: Vec<bool> = hits.iter().map(|_| REJECT_COPY_ITEMS.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1)).is_ok()).collect();
        let request=internal("POST",format_args!("/_bulk"),Vec::new(),true).and_then(|mut req|{crate::front::bypass_pipelines(&mut req, routings)?;req.try_with_encoded_body(|out| {
            #[derive(serde::Serialize)] struct Meta<'a>{_index:&'a str,_id:&'a str,#[serde(skip_serializing_if="Option::is_none")]routing:Option<&'a str>,#[serde(skip_serializing_if="Option::is_none")]version:Option<u64>,#[serde(skip_serializing_if="Option::is_none")]version_type:Option<&'static str>}
            #[derive(serde::Serialize)] struct Action<'a>{index:Meta<'a>}
            for (h, reject) in hits.iter().zip(&rejected) {
                let Some(id)=h["_id"].as_str() else{continue};let version=h["_version"].as_u64();
                serde_json::to_writer(&mut *out,&Action{index:Meta{_index:target,_id:id,routing:h["_routing"].as_str(),version,version_type:version.map(|_|"external")}})?;out.write_all(b"\n")?;
                if *reject {
                    out.write_all(br#"{"_id":"rejected-by-test-seam"}"#)?;
                } else {
                    serde_json::to_writer(&mut *out,&h["_source"])?;
                }
                out.write_all(b"\n")?;
            } Ok(())
        })});
        let r=send(node,request).await;
        if !r.is_success() {
            return Err(format!("bulk into [{target}] answered {}", r.status));
        }
        let items = r.body_json().map(|v| v["items"].as_array().cloned().unwrap_or_default()).unwrap_or_default();
        if items.len() != routings_len(hits) {
            return Err(format!("bulk into [{target}] answered {} items for {} documents", items.len(), routings_len(hits)));
        }
        for it in &items {
            match it["index"]["status"].as_u64().unwrap_or(0) {
                200 | 201 => *copied += 1,
                // Already present with this version (copied by the engine);
                // the verification checks it is the same document.
                409 => {}
                s => return Err(format!("bulk item into [{target}] answered {s}: {}", it["index"]["error"]["reason"])),
            }
        }
    }
}

fn routings_len(hits: &[Value]) -> usize {
    hits.iter().filter(|h| h["_id"].is_string()).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sub_requests_carry_the_callers_credentials() {
        let creds = crate::front::SharedHeaders::try_new(&[("authorization".to_string(), "Basic dTpw".to_string())]).unwrap();
        let sub = CREDENTIALS.scope(creds, async { internal("POST", format_args!("/_bulk"), vec![], true) }).await.unwrap();
        assert!(sub.headers.iter().any(|(k, v)| k == "authorization" && v == "Basic dTpw"));
        assert!(sub.headers.iter().any(|(k, v)| k == "content-type" && v == "application/x-ndjson"));
        // Outside a resize: no credentials.
        assert!(!internal("GET", format_args!("/x"), vec![], false).unwrap().headers.iter().any(|(k, _)| k == "authorization"));
    }
}
