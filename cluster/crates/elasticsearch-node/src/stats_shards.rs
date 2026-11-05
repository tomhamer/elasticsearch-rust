//! `_shards` of `GET [/{index}]/_stats` under the cluster.
//!
//! A stats request is answered by one node's engine, which counts one
//! shard per index (the cluster runs every index as a single primary plus
//! replicas, see `mirror::extract`). ES counts every shard copy the index
//! was created with: `number_of_shards * (1 + number_of_replicas)`. The
//! header is recomputed here from the index settings (which keep the
//! requested counts) and the routing table.

use crate::front::{RestReq, RestResp};
use crate::node::ClusterNode;
use elasticsearch_engine::node::ResolveOpts;
use serde_json::{json, Value};
use std::sync::Arc;

fn setting_u64(settings: &serde_json::Map<String, Value>, key: &str, default: u64) -> u64 {
    settings.get(key).and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).unwrap_or(default)
}

/// Rewrite the top-level `_shards` of a successful stats response.
pub async fn adjust(node: &Arc<ClusterNode>, req: &RestReq, targets: &str, resp: RestResp) -> RestResp {
    adjust_resolved(node, req, targets, None, resp).await
}

/// [`adjust`] with `targets` already resolved to `names` (`None`: resolve
/// them here).
pub async fn adjust_resolved(node: &Arc<ClusterNode>, req: &RestReq, targets: &str, names: Option<std::collections::BTreeSet<String>>, resp: RestResp) -> RestResp {
    if !resp.is_success() {
        return resp;
    }
    let t0 = std::time::Instant::now();
    let Some(mut body) = resp.body_json() else { return resp };
    if !body["_shards"]["total"].is_u64() {
        return resp;
    }
    let parse_us = t0.elapsed().as_micros();
    let names = match names {
        Some(names) => names,
        None => {
            let opts = ResolveOpts::from_params(&req.params());
            let engine = node.engine.clone();
            let expr = targets.to_string();
            let resolved = tokio::task::spawn_blocking(move || engine.resolve(&expr, opts).map(|ts| ts.iter().map(|t| t.svc.name.clone()).collect::<std::collections::BTreeSet<_>>())).await;
            let Ok(Ok(names)) = resolved else { return resp };
            names
        }
    };
    let st = node.state();
    // Metadata exists even when analyzer recovery prevented every shard
    // from opening. Index stats are a broadcast response from active copies.
    if let Some(indices) = body.get_mut("indices").and_then(Value::as_object_mut) {
        indices.retain(|name, _| st.routing_table.all_shards().any(|copy| copy.index == *name && copy.active()));
    }
    let (mut total, mut successful) = (0u64, 0u64);
    // Indices whose `total` sums several copies' engines.
    let mut multi = Vec::new();
    let mut all_shards = Vec::new();
    for name in &names {
        let Some(im) = st.metadata.indices.get(name) else { continue };
        let shards = setting_u64(&im.settings, "index.number_of_shards", 1).max(1);
        let copies = 1 + setting_u64(&im.settings, "index.number_of_replicas", u64::from(im.number_of_replicas));
        total += shards * copies;
        if st.routing_table.shard_table(name, 1).is_some() {
            // Every shard has its own table (and engine).
            for k in 0..shards as u32 {
                successful += st.routing_table.shard_table(name, k).map(|t| t.active_copies().count() as u64).unwrap_or(0).min(copies);
            }
            multi.push((name.clone(), shards as u32));
        } else {
            let active = st.routing_table.shard_table(name, 0).map(|t| t.active_copies().count() as u64).unwrap_or(0);
            successful += shards * active.min(copies);
            // The answering engine measured one copy; ES's `total` sums
            // every active copy.
            if active > 1 {
                multi.push((name.clone(), 1));
            }
        }
        all_shards.push((name.clone(), if st.routing_table.shard_table(name, 1).is_some() { shards as u32 } else { 1 }));
    }
    // `level=shards` lists every active copy of every shard, each with its
    // own stats and routing (ES `ShardStats` per copy, from its node).
    let level_shards = body["indices"].as_object().is_some_and(|m| m.values().any(|v| v["shards"].is_object()));
    // Only each copy's document counts and store size are read here.
    let gather_at = t0.elapsed().as_micros();
    let stats = if level_shards || !multi.is_empty() { match crate::rest::gather_shard_doc_stats(node, &st).await {Ok(v)=>v,Err(e)=>return RestResp::from_es_error(&e)} } else { Default::default() };
    let gathered_at = t0.elapsed().as_micros();
    if level_shards {
        for (name, shards) in &all_shards {
            if let Err(r)=render_copies(node, req, &st, name, *shards, &stats, &mut body).await { return r; }
        }
    }
    // The answering engine only sees one copy of shard 0: the document
    // counts of a multi-shard (or replicated) index come from every node's
    // shard copies instead.
    if !multi.is_empty() && body["indices"].is_object() {
        for (name, shards) in &multi {
            let (mut pri, mut tot) = ((0u64, 0u64, 0u64), (0u64, 0u64, 0u64));
            for k in 0..*shards {
                let Some(t) = st.routing_table.shard_table(name, k) else { continue };
                for c in t.active_copies() {
                    let Some(s) = c.allocation_id().and_then(|a| stats.get(a)) else { continue };
                    tot.0 += s.docs;
                    tot.1 += s.deleted;
                    tot.2 += s.store_bytes;
                    if c.primary {
                        pri.0 += s.docs;
                        pri.1 += s.deleted;
                        pri.2 += s.store_bytes;
                    }
                }
            }
            for (section, (count, deleted, store)) in [("primaries", pri), ("total", tot)] {
                let docs = &body["indices"][name.as_str()][section]["docs"];
                if docs.is_object() {
                    let (old_c, old_d) = (docs["count"].as_u64().unwrap_or(0), docs["deleted"].as_u64().unwrap_or(0));
                    body["indices"][name.as_str()][section]["docs"]["count"] = json!(count);
                    body["indices"][name.as_str()][section]["docs"]["deleted"] = json!(deleted);
                    // Live-doc size, like store, was measured on shard 0 only.
                    let old_sz = docs_size(&mut body["indices"][name.as_str()][section]["docs"], None, store);
                    let all = &mut body["_all"][section]["docs"];
                    if all.is_object() {
                        let c = all["count"].as_u64().unwrap_or(0) + count;
                        let d = all["deleted"].as_u64().unwrap_or(0) + deleted;
                        all["count"] = json!(c.saturating_sub(old_c));
                        all["deleted"] = json!(d.saturating_sub(old_d));
                        docs_size(all, old_sz, store);
                    }
                }
                // Store size: the answering engine measured shard 0 only.
                if body["indices"][name.as_str()][section]["store"].is_object() {
                    let old = set_store(&mut body["indices"][name.as_str()][section]["store"], store);
                    let all = &mut body["_all"][section]["store"];
                    if all.is_object() {
                        let cur = all["size_in_bytes"].as_u64().unwrap_or(0);
                        set_store(all, (cur + store).saturating_sub(old));
                    }
                }
            }
            if let Err(r)=sum_segment_sections(node, req, &st, name, *shards, &mut body).await { return r; }
        }
    }
    body["_shards"]["total"] = json!(total);
    body["_shards"]["successful"] = json!(successful);
    body["_shards"]["failed"] = json!(0);
    let sections_at = t0.elapsed().as_micros();
    let mut out = RestResp::json(resp.status, &body);
    crate::front::stats_phases::record("index_stats_adjust", vec![("parse", parse_us), ("resolve", gather_at - parse_us), ("doc_stats", gathered_at - gather_at),
        ("sections", sections_at - gathered_at), ("render", t0.elapsed().as_micros() - sections_at)]);
    for (k, v) in &resp.headers {
        if !k.eq_ignore_ascii_case("content-type") && !k.eq_ignore_ascii_case("content-length") {
            out.push_header((k.clone(),v.clone()));
        }
    }
    out
}

/// Replace `docs.total_size_in_bytes` (when the engine reports it) with
/// `bytes`; `_all` (`prev` = the index's old value) moves by the difference.
/// Returns the old value.
fn docs_size(docs: &mut Value, prev: Option<u64>, bytes: u64) -> Option<u64> {
    let cur = docs.get("total_size_in_bytes")?.as_u64()?;
    let new = match prev {
        Some(old) => (cur + bytes).saturating_sub(old),
        None => bytes,
    };
    docs["total_size_in_bytes"] = json!(new);
    Some(cur)
}

/// The per-shard sections (all but `docs` / `store`, summed above from the
/// copies' shard stats, and `mappings`, which is per index) of a
/// multi-shard index: the answering engine only measured its shard 0, so
/// every active copy's engine is asked for its own and the results summed
/// (primaries from primary copies, total from all; ES sums `CommonStats`
/// over shard copies), `_all` moving by the difference.
async fn sum_segment_sections(node: &Arc<ClusterNode>, req: &RestReq, st: &elasticsearch_cluster::state::ClusterState, name: &str, shards: u32, body: &mut Value) -> Result<(),RestResp> {
    const SECTIONS: [&str; 18] = [
        "completion", "fielddata", "indexing", "get", "search", "merges", "refresh", "flush", "warmer", "query_cache", "segments", "translog", "request_cache", "recovery", "bulk",
        "shard_stats", "dense_vector", "sparse_vector",
    ];
    let present: Vec<&str> = SECTIONS.iter().copied().filter(|s| body["indices"][name]["primaries"][*s].is_object() || body["indices"][name]["total"][*s].is_object()).collect();
    if present.is_empty() {
        return Ok(());
    }
    let mut sub = req.clone();
    let q = req.query();
    sub.set_method("GET").and_then(|_|sub.set_body(&[])).map_err(|e|RestResp::json(e.status,&e.to_json()))?;
    // Engine metric names (`merges` is the `merge` metric).
    let engine_metrics: Vec<&str> = present.iter().filter(|s| !matches!(**s, "global_ordinals" | "indexing_pressure")).map(|s| if *s == "merges" { "merge" } else { *s }).collect();
    let metrics = crate::front::UriJoin(engine_metrics.iter().copied(), ",");
    sub.set_uri_fmt(format_args!("/{}/_stats/{}{}{}", name, metrics, if q.is_empty() { "" } else { "?" }, q)).map_err(|e|RestResp::json(e.status,&e.to_json()))?;
    // This node's copies are measured in process on their shard engines
    // (`local_copy_sections`); other nodes' copies are asked over the
    // transport. Answers are summed in copy order either way.
    let mut local = Vec::new();
    let mut futs = Vec::new();
    let mut seq = 0usize;
    for k in 0..shards {
        let Some(t) = st.routing_table.shard_table(name, k) else { continue };
        for c in t.active_copies() {
            let Some(n) = c.node_id.clone() else { continue };
            seq += 1;
            if n == node.local.id {
                local.push((seq, k, c.primary));
                continue;
            }
            let (sub, node, primary) = (sub.clone(), node.clone(), c.primary);
            futs.push(async move { (seq, primary, crate::front::exec_shard(&node, &n, &sub, k).await) });
        }
    }
    let local_job = local_copy_sections(node, name, engine_metrics.join(","), req.params(), local);
    let (local_answers, remote) = futures::future::join(local_job, futures::future::join_all(futs)).await;
    let mut answers = local_answers?;
    for (seq, primary, r) in remote {
        if r.status==429 { return Err(r); }
        let Some(b) = r.body_json().filter(|_| r.is_success()) else { continue };
        answers.push((seq, primary, b));
    }
    answers.sort_by_key(|(seq, _, _)| *seq);
    let (mut pri, mut tot) = (serde_json::Map::new(), serde_json::Map::new());
    for (_, primary, b) in &answers {
        let primary = *primary;
        for s in &present {
            // A copy's engine runs it as its only (primary) shard.
            let v = &b["indices"][name]["primaries"][*s];
            if !v.is_object() {
                continue;
            }
            let mut targets = vec![&mut tot];
            if primary {
                targets.push(&mut pri);
            }
            for m in targets {
                match m.get_mut(*s) {
                    Some(cur) => add_json(cur, v),
                    None => {
                        m.insert((*s).to_string(), v.clone());
                    }
                }
            }
        }
    }
    for (section, sums) in [("primaries", pri), ("total", tot)] {
        for s in &present {
            if !body["indices"][name][section][*s].is_object() {
                continue;
            }
            let mut new = sums.get(*s).cloned().unwrap_or_else(|| zeroed(&body["indices"][name][section][*s]));
            fix_human(&mut new);
            let old = std::mem::replace(&mut body["indices"][name][section][*s], new.clone());
            let all = &mut body["_all"][section][*s];
            if all.is_object() {
                sub_json(all, &old);
                add_json(all, &new);
                fix_human(all);
            }
        }
    }
    Ok(())
}

/// The stats of this node's copies of `name` in `copies` (`(order, shard,
/// primary)`) for the engine `metrics`, each measured in process on its
/// shard engine: the body `exec_shard` asked that engine's route for,
/// without an in-process HTTP round trip per copy (request admission,
/// security, usage counting, an executor hop, serialization, the uuid
/// rewrite scan and a re-parse). `params` is the request's query, so the
/// copies render exactly what the routed sub-request did (`level`, `human`,
/// index options). A copy whose engine or index is gone answers nothing,
/// as its failed sub-request did; a breaker trip fails the request.
async fn local_copy_sections(node: &Arc<ClusterNode>, name: &str, metrics: String, params: std::collections::HashMap<String, String>, copies: Vec<(usize, u32, bool)>) -> Result<Vec<(usize, bool, Value)>, RestResp> {
    if copies.is_empty() {
        return Ok(Vec::new());
    }
    let jobs: Vec<(usize, bool, Arc<elasticsearch_engine::node::Node>)> = copies.into_iter().filter_map(|(seq, k, primary)| node.engine_for(k).map(|engine| (seq, primary, engine))).collect();
    // Each copy's engine computes concurrently (one blocking job each; one
    // after another they added up across a multi-shard index's copies).
    let handles: Vec<_> = jobs.into_iter().map(|(seq, primary, engine)| {
        let (index, metrics, params) = (name.to_string(), metrics.clone(), params.clone());
        let job = elasticsearch_engine::memory::inherit_request_policy(move || elasticsearch_http::index_stats_in_process(&engine, &index, Some(metrics.as_str()), params));
        (seq, primary, tokio::task::spawn_blocking(job))
    }).collect();
    let mut done = Vec::with_capacity(handles.len());
    for (seq, primary, handle) in handles {
        let r = handle.await.map_err(|e| RestResp::error(500, "exception", &format!("shard copy stats of [{name}] failed: {e}")))?;
        done.push((seq, primary, r));
    }
    let mut out = Vec::with_capacity(done.len());
    for (seq, primary, r) in done {
        match r {
            Ok(v) => out.push((seq, primary, v)),
            Err(e) if e.status == 429 => return Err(RestResp::from_es_error(&e)),
            Err(_) => {}
        }
    }
    Ok(out)
}

/// `v` with every numeric leaf set to 0.
fn zeroed(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().map(|(k, v)| (k.clone(), zeroed(v))).collect()),
        Value::Number(_) => json!(0),
        o => o.clone(),
    }
}

/// Subtract the numeric leaves of `src` from `dst` (saturating at 0);
/// object keys whose subtree ends up all-zero and were only in `src` are
/// left alone.
/// Stats leaves that are not sums over shard copies.
const NOT_SUMMED: &[&str] = &["max_unsafe_auto_id_timestamp"];

fn sub_json(dst: &mut Value, src: &Value) {
    match (dst, src) {
        (Value::Object(d), Value::Object(s)) => {
            for (k, v) in s {
                if NOT_SUMMED.contains(&k.as_str()) {
                    continue;
                }
                if let Some(cur) = d.get_mut(k) {
                    sub_json(cur, v);
                }
            }
        }
        (d @ Value::Number(_), Value::Number(s)) => {
            if let (Some(a), Some(b)) = (d.as_u64(), s.as_u64()) {
                *d = json!(a.saturating_sub(b));
            }
        }
        _ => {}
    }
}

/// Set a `store` stats object's sizes to `bytes` (keeping the `size`
/// string of `?human` responses in step); returns the previous size.
/// The `level=shards` entries of `name`: every active copy of each shard
/// renders its own entry on its node's engine (which runs the copy as its
/// shard "0"), labelled with the copy's routing. A copy that does not
/// answer keeps no entry (ES counts it as a failed shard).
async fn render_copies(node: &Arc<ClusterNode>, req: &RestReq, st: &elasticsearch_cluster::state::ClusterState, name: &str, shards: u32, stats: &std::collections::BTreeMap<crate::front::SharedText, crate::rest::LocalShardStats>, body: &mut Value) -> Result<(),RestResp> {
    if !body["indices"][name]["shards"].is_object() {
        return Ok(());
    }
    // The requested metrics (`/_stats/{metric}` or `/{index}/_stats/{metric}`).
    let segs = req.segments();
    let metric = segs.iter().position(|s| s == "_stats").and_then(|i| segs.get(i + 1)).map(String::as_str);
    let metrics = metric.unwrap_or("");
    let mut sub = req.clone();
    let q = req.query();
    sub.set_method("GET").and_then(|_|sub.set_body(&[])).map_err(|e|RestResp::json(e.status,&e.to_json()))?;
    sub.set_uri_fmt(format_args!("/{name}/_stats{}{metrics}{}{q}", if metric.is_some() { "/" } else { "" }, if q.is_empty() { "" } else { "?" })).map_err(|e|RestResp::json(e.status,&e.to_json()))?;
    let mut futs = Vec::new();
    for k in 0..shards {
        let Some(t) = st.routing_table.shard_table(name, k) else { continue };
        let mut copies: Vec<_> = t.active_copies().filter(|c| c.node_id.is_some()).cloned().collect();
        copies.sort_by_key(|c| !c.primary);
        for c in copies {
            let (sub, node) = (sub.clone(), node.clone());
            futs.push(async move {
                let r = crate::front::exec_shard(&node, c.node_id.as_deref().unwrap_or_default(), &sub, k).await;
                (k, c, r)
            });
        }
    }
    let mut out: std::collections::BTreeMap<u32, Vec<Value>> = Default::default();
    for (k, c, r) in futures::future::join_all(futs).await {
        if r.status==429 { return Err(r); }
        let Some(b) = r.body_json().filter(|_|r.is_success()) else { continue };
        let Some(mut e) = b["indices"][name]["shards"].as_object().and_then(|m| m.values().next()).and_then(|v| v.as_array()).and_then(|a| a.first()).cloned() else { continue };
        if e["routing"].is_object() {
            e["routing"]["state"] = json!(if c.state == elasticsearch_cluster::state::ShardState::Relocating { "RELOCATING" } else { "STARTED" });
            e["routing"]["relocating_node"] = json!(c.relocating_node_id);
        }
        match c.allocation_id().and_then(|a| stats.get(a)) {
            Some(s) => fill_copy(&mut e, c.primary, s, c.node_id.clone()),
            None => {
                if e["routing"].is_object() {
                    e["routing"]["primary"] = json!(c.primary);
                    e["routing"]["node"] = json!(c.node_id);
                }
            }
        }
        out.entry(k).or_default().push(e);
    }
    for (k, entries) in out {
        body["indices"][name]["shards"][k.to_string()] = Value::Array(entries);
    }
    Ok(())
}

/// Labels one `level=shards` entry with a shard copy's role, node and stats.
fn fill_copy(e: &mut Value, primary: bool, s: &crate::rest::LocalShardStats, node_id: Option<String>) {
    if e["routing"].is_object() {
        e["routing"]["primary"] = json!(primary);
        e["routing"]["node"] = json!(node_id);
    }
    if e["store"].is_object() {
        set_store(&mut e["store"], s.store_bytes);
    }
    if e["docs"].is_object() {
        e["docs"]["count"] = json!(s.docs);
        e["docs"]["deleted"] = json!(s.deleted);
    }
}

pub(crate) fn set_store(store: &mut Value, bytes: u64) -> u64 {
    let old = store["size_in_bytes"].as_u64().unwrap_or(0);
    store["size_in_bytes"] = json!(bytes);
    if store.get("total_data_set_size_in_bytes").is_some() {
        store["total_data_set_size_in_bytes"] = json!(bytes);
    }
    if store.get("size").is_some() {
        store["size"] = json!(human_bytes(bytes));
    }
    if store.get("total_data_set_size").is_some() {
        store["total_data_set_size"] = json!(human_bytes(bytes));
    }
    old
}

/// Add the numeric leaves of `src` into `dst` (missing keys are copied).
pub(crate) fn add_json(dst: &mut Value, src: &Value) {
    match (dst, src) {
        (Value::Object(d), Value::Object(s)) => {
            for (k, v) in s {
                if NOT_SUMMED.contains(&k.as_str()) {
                    continue;
                }
                match d.get_mut(k) {
                    Some(cur) => add_json(cur, v),
                    None => {
                        d.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (d @ Value::Number(_), Value::Number(s)) => {
            if let (Some(a), Some(b)) = (d.as_u64(), s.as_u64()) {
                *d = json!(a + b);
            } else if let (Some(a), Some(b)) = (d.as_f64(), s.as_f64()) {
                *d = json!(a + b);
            }
        }
        _ => {}
    }
}

/// Re-render every `k` string whose sibling `k_in_bytes` is a number.
pub(crate) fn fix_human(v: &mut Value) {
    let Value::Object(m) = v else { return };
    let keys: Vec<String> = m.keys().cloned().collect();
    let mut stale = false;
    for k in &keys {
        if let Some(b) = m.get(&format!("{k}_in_bytes")).and_then(Value::as_u64) {
            if m[k].is_string() {
                m.insert(k.clone(), json!(human_bytes(b)));
            }
        }
        // Times (`total_time` of `total_time_in_millis`, ...): re-rendered
        // by the engine's `?human` below.
        if ["_in_millis", "_in_nanos"].iter().any(|suf| m.get(&format!("{k}{suf}")).is_some_and(Value::is_number)) && m[k].is_string() {
            m.remove(k);
            stale = true;
        }
    }
    for (_, child) in m.iter_mut() {
        fix_human(child);
    }
    if stale {
        elasticsearch_http::humanize(v);
    }
}

/// ES `ByteSizeValue` rendering (`1.2kb`, `512b`; truncated like ES).
pub(crate) fn human_bytes(b: u64) -> String {
    elasticsearch_engine::memory::byte_size(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_matches_es() {
        assert_eq!(human_bytes(512), "512b");
        assert_eq!(human_bytes(1024), "1kb");
        assert_eq!(human_bytes(1536), "1.5kb");
        assert_eq!(human_bytes(3 * (1 << 20)), "3mb");
    }

    #[test]
    fn set_store_updates_sizes() {
        let mut s = json!({"size_in_bytes": 10, "total_data_set_size_in_bytes": 10, "reserved_in_bytes": 0});
        assert_eq!(set_store(&mut s, 2048), 10);
        assert_eq!(s, json!({"size_in_bytes": 2048, "total_data_set_size_in_bytes": 2048, "reserved_in_bytes": 0}));
    }
}
