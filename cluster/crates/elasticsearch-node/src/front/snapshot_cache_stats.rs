//! Searchable snapshot statistics on a cluster:
//! * `GET /_searchable_snapshots[/{node_id}]/cache/stats`: one entry per
//!   selected node, keyed by its node id (each engine reports only itself,
//!   under its engine-local id);
//! * `GET [/{index}]/_searchable_snapshots/stats`: the shard file
//!   statistics of the node holding the mounted copies.
use super::*;

/// The node selector of the request (`_all` without one).
fn selector(segs: &[String]) -> Option<String> {
    match segs.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["_searchable_snapshots", "cache", "stats"] => Some("_all".to_string()),
        ["_searchable_snapshots", n, "cache", "stats"] => Some((*n).to_string()),
        _ => None,
    }
}

/// The single node entry of an engine's response.
fn only_entry(v: &Value) -> Option<Value> {
    v["nodes"].as_object()?.values().next().cloned()
}

pub(super) async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> Option<RestResp> {
    let spec = selector(&req.segments())?;
    if req.method() != Method::GET {
        return None;
    }
    // Parameter errors exactly as the engine renders them.
    let local_uri = format!("/_searchable_snapshots/cache/stats{}", req.uri.split_once('?').map(|(_, q)| format!("?{q}")).unwrap_or_default());
    let mut probe_req = req.clone();
    if let Err(error) = probe_req.set_uri(&local_uri) { return Some(RestResp::json(error.status, &error.to_json())); }
    let probe = local_canonical(node, &probe_req).await;
    if !probe.is_success() {
        return Some(probe);
    }
    let st = node.state();
    let ids = crate::rest::nodes::resolve_nodes(&st, &node.local.id, &spec);
    let mut nodes = serde_json::Map::new();
    for id in ids {
        let r = if id == node.local.id {
            probe.clone()
        } else {
            let mut sub = req.clone();
            if let Err(error) = sub.set_uri(&local_uri) { return Some(RestResp::json(error.status, &error.to_json())); }
            if let Err(error) = sub.push_header(EXEC_LOCAL, "1") { return Some(RestResp::json(error.status, &error.to_json())); }
            forward(node, &id, &sub).await
        };
        if let Some(entry) = r.body_json().as_ref().filter(|_| r.is_success()).and_then(only_entry) {
            nodes.insert(id, entry);
        }
    }
    Some(RestResp::json(200, &json!({"nodes": nodes})))
}

/// The index expression of a `[/{index}]/_searchable_snapshots/stats` request.
fn stats_target(segs: &[String]) -> Option<String> {
    match segs.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["_searchable_snapshots", "stats"] => Some("_all".to_string()),
        [index, "_searchable_snapshots", "stats"] => Some((*index).to_string()),
        _ => None,
    }
}

/// `[/{index}]/_searchable_snapshots/stats`: engines report the shards they
/// hold, so the request runs on the one node holding every mounted copy it
/// targets (mounted indices have no replicas), with the shards' node ids.
pub(super) async fn handle_stats(node: &Arc<ClusterNode>, req: &RestReq) -> Option<RestResp> {
    let target = stats_target(&req.segments())?;
    if req.method() != Method::GET {
        return None;
    }
    let engine = node.engine.clone();
    let opts = ResolveOpts::from_params(&req.params());
    let names: Vec<String> = tokio::task::spawn_blocking(move || {
        engine.resolve(&target, opts).map(|ts| ts.iter().filter(|t| elasticsearch_engine::snapshot::is_searchable_snapshot(&t.svc)).map(|t| t.svc.name.clone()).collect()).unwrap_or_default()
    })
    .await
    .unwrap_or_default();
    let st = node.state();
    let mut holders: BTreeSet<String> = BTreeSet::new();
    for name in &names {
        let n = st.metadata.indices.get(name).map(|m| m.number_of_shards.max(1)).unwrap_or(1);
        for k in 0..n {
            if let Some(id) = st.routing_table.shard_table(name, k).and_then(|t| t.active_primary()).and_then(|c| c.node_id.clone()) {
                holders.insert(id);
            }
        }
    }
    let holder = match holders.len() {
        1 => holders.into_iter().next()?,
        _ => return None,
    };
    let r = if holder == node.local.id {
        local_canonical(node, req).await
    } else {
        let mut sub = req.clone();
        if let Err(error) = sub.push_header(EXEC_LOCAL, "1") { return Some(RestResp::json(error.status, &error.to_json())); }
        forward(node, &holder, &sub).await
    };
    Some(with_node_id(r, &holder))
}

/// The engine's local node id in a response, as the cluster's node id.
fn with_node_id(mut r: RestResp, id: &str) -> RestResp {
    let engine_id = format!("\"{}\"", elasticsearch_engine::node::NODE_ID);
    if let Ok(text) = std::str::from_utf8(&r.body).map(str::to_owned) {
        r.replace_body(text.replace(&engine_id, &format!("\"{id}\"")).into_bytes());
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_and_entry() {
        let s = |p: &str| selector(&p.split('/').filter(|x| !x.is_empty()).map(str::to_string).collect::<Vec<_>>());
        assert_eq!(s("/_searchable_snapshots/cache/stats").as_deref(), Some("_all"));
        assert_eq!(s("/_searchable_snapshots/n1/cache/stats").as_deref(), Some("n1"));
        assert_eq!(s("/_searchable_snapshots/stats"), None);
        assert_eq!(only_entry(&json!({"nodes": {"x": {"shared_cache": {}}}})), Some(json!({"shared_cache": {}})));
        let t = |p: &str| stats_target(&p.split('/').filter(|x| !x.is_empty()).map(str::to_string).collect::<Vec<_>>());
        assert_eq!(t("/_searchable_snapshots/stats").as_deref(), Some("_all"));
        assert_eq!(t("/d*/_searchable_snapshots/stats").as_deref(), Some("d*"));
        let r = with_node_id(RestResp::json(200, &json!({"node": elasticsearch_engine::node::NODE_ID})), "n1");
        assert_eq!(r.body_json().unwrap()["node"], "n1");
    }
}
