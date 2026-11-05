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

use crate::front::{dispatch, on_master_route, RestReq, RestResp};
use crate::node::ClusterNode;
use axum::http::Method;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

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

async fn wait_for_primary(node: &ClusterNode, index: &str, timeout: Duration) -> bool {
    let mut rx = node.subscribe();
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        // Every target shard: the copy bulk-routes documents across all of
        // them (a split / shrink target has its own shard count).
        let ok = {
            let st = rx.borrow();
            let n = crate::shard_engines::shards_of(&st.metadata, index).max(1);
            st.metadata.indices.contains_key(index) && (0..n).all(|k| st.routing_table.shard_table(index, k).and_then(|t| t.active_primary()).is_some())
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
    if !wait_for_primary(node, target, Duration::from_secs(30)).await {
        tracing::warn!(source, target, "resize: target primary did not start; documents not copied");
        return resp;
    }
    // The target inherits the source's write block (a resize source must be
    // read-only); lift it for the copy and put it back afterwards.
    let prepared_call = call(node, "GET", format_args!("/{target}/_settings?flat_settings=true"), None);
    let blocked = prepared_call.await
        .body_json()
        .map(|v| {
            let b = &v[target]["settings"]["index.blocks.write"];
            b == &json!("true") || b == &json!(true)
        })
        .unwrap_or(false);
    if blocked {
        let prepared_call = call(node, "PUT", format_args!("/{target}/_settings"), Some(json!({"index.blocks.write": false})));
        let r = prepared_call.await;
        if !r.is_success() {
            tracing::warn!(source, target, status = r.status, "resize: could not lift the target's write block; documents not copied");
            return resp;
        }
    }
    match copy_docs(node, source, target).await {
        Ok((n, misplaced)) => {
            tracing::debug!(source, target, copied = n, "resize: documents copied to the target primary");
            if let Err(e) = purge_misplaced(node, target, &misplaced).await {
                tracing::warn!(source, target, error = %e, "resize: removing documents of other shards from target shard 0 failed");
            }
        }
        Err(e) => tracing::warn!(source, target, error = %e, "resize: copying documents to the target failed"),
    }
    if blocked {
        let prepared_call = call(node, "PUT", format_args!("/{target}/_settings"), Some(json!({"index.blocks.write": true})));
        let r = prepared_call.await;
        if !r.is_success() {
            tracing::warn!(source, target, status = r.status, "resize: could not restore the target's write block");
        }
    }
    resp
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
/// local target shards.
async fn copy_docs(node: &Arc<ClusterNode>, source: &str, target: &str) -> Result<(u64, Vec<String>), String> {
    let md = node.state().metadata.clone();
    let multi = crate::shard_engines::shards_of(&md, target) > 1;
    let (mut elsewhere, mut in_zero) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
    // The source is write-blocked; make everything it holds searchable.
    let prepared_call = call(node, "POST", format_args!("/{source}/_refresh"), None);
    let _ = prepared_call.await;
    let prepared_call = call(node, "POST", format_args!("/{source}/_search?scroll=1m&size={PAGE}&expand_wildcards=all"), Some(json!({"version": true, "sort": ["_doc"]})));
    let first = prepared_call.await;
    if !first.is_success() {
        return Err(format!("search of [{source}] answered {}", first.status));
    }
    let mut page = first.body_json().unwrap_or(Value::Null);
    let mut copied = 0u64;
    let mut scroll_id: Option<String> = None;
    let result = loop {
        if let Some(id) = page["_scroll_id"].as_str() {
            scroll_id = Some(id.to_string());
        }
        let hits = page["hits"]["hits"].as_array().map(Vec::as_slice).unwrap_or_default();
        if hits.is_empty() {
            break Ok(copied);
        }
        for h in hits {
            let Some(id) = h["_id"].as_str() else { continue };
            if multi {
                match crate::shard_engines::doc_shard(&md, target, id, h["_routing"].as_str()) {
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
        let request=internal("POST",format_args!("/_bulk"),Vec::new(),true).and_then(|req|req.try_with_encoded_body(|out| {
            #[derive(serde::Serialize)] struct Meta<'a>{_index:&'a str,_id:&'a str,#[serde(skip_serializing_if="Option::is_none")]routing:Option<&'a str>,#[serde(skip_serializing_if="Option::is_none")]version:Option<u64>,#[serde(skip_serializing_if="Option::is_none")]version_type:Option<&'static str>}
            #[derive(serde::Serialize)] struct Action<'a>{index:Meta<'a>}
            for h in hits {
                let Some(id)=h["_id"].as_str() else{continue};let version=h["_version"].as_u64();
                serde_json::to_writer(&mut *out,&Action{index:Meta{_index:target,_id:id,routing:h["_routing"].as_str(),version,version_type:version.map(|_|"external")}})?;out.write_all(b"\n")?;
                serde_json::to_writer(&mut *out,&h["_source"])?;out.write_all(b"\n")?;
            } Ok(())
        }));
        let r=send(node,request).await;
        if !r.is_success() {
            break Err(format!("bulk into [{target}] answered {}", r.status));
        }
        let items = r.body_json().map(|v| v["items"].as_array().cloned().unwrap_or_default()).unwrap_or_default();
        let mut failed = None;
        for it in &items {
            match it["index"]["status"].as_u64().unwrap_or(0) {
                200 | 201 => copied += 1,
                // Already present with this version (copied by the engine).
                409 => {}
                s => failed = Some(format!("bulk item into [{target}] answered {s}: {}", it["index"]["error"]["reason"])),
            }
        }
        if let Some(e) = failed {
            break Err(e);
        }
        let Some(id) = scroll_id.clone() else { break Ok(copied) };
        let prepared_call = call(node, "POST", format_args!("/_search/scroll"), Some(json!({"scroll": "1m", "scroll_id": id})));
        let next = prepared_call.await;
        if !next.is_success() {
            break Err(format!("scroll of [{source}] answered {}", next.status));
        }
        page = next.body_json().unwrap_or(Value::Null);
    };
    if let Some(id) = scroll_id {
        let prepared_call = call(node, "DELETE", format_args!("/_search/scroll"), Some(json!({"scroll_id": id})));
        let _ = prepared_call.await;
    }
    let prepared_call = call(node, "POST", format_args!("/{target}/_refresh"), None);
    let _ = prepared_call.await;
    // An id that also routes to shard 0 (under another routing) stays.
    result.map(|n| (n, elsewhere.difference(&in_zero).cloned().collect()))
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
