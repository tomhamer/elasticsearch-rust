//! Execute application queries on physical shard copies while retaining their
//! original REST workflow and authorization on every executing node.
use super::*;
use std::collections::BTreeMap;

pub(super) fn is_search(request: &RestReq) -> bool {
    let parts = request.segments();
    parts.len() == 4 && parts[0] == "_application" && parts[1] == "search_application"
        && parts[3] == "_search" && matches!(request.method(), Method::GET | Method::POST)
}

pub(super) fn execute<'a>(node: &'a Arc<ClusterNode>, request: &'a RestReq, indices: &'a [String], query: &'a Value, params: &'a HashMap<String, String>) -> futures::future::BoxFuture<'a, RestResp> {
    Box::pin(execute_inner(node, request, indices, query, params))
}

fn candidates(state: &elasticsearch_cluster::state::ClusterState, indices: &[String], shard: u32, preferred: &str) -> Vec<String> {
    let mut common: Option<BTreeSet<String>> = None;
    for index in indices {
        let owners: BTreeSet<String> = state.routing_table.shard_table(index, shard).into_iter()
            .flat_map(|table| table.searchable_copies(&state.nodes)).filter_map(|copy| copy.node_id.clone()).collect();
        common = Some(match common { None => owners, Some(previous) => previous.intersection(&owners).cloned().collect() });
    }
    let mut result = vec![preferred.to_string()];
    result.extend(common.unwrap_or_default().into_iter().filter(|owner| owner != preferred));
    result
}

async fn execute_inner(node: &Arc<ClusterNode>, request: &RestReq, indices: &[String], query: &Value, params: &HashMap<String, String>) -> RestResp {
    await_mirror(node).await;
    let state = node.state();
    let mut groups: BTreeMap<(String, u32), Vec<String>> = BTreeMap::new();
    for index in indices {
        if !state.metadata.indices.contains_key(index) {
            let error = elasticsearch_engine::error::EsError::index_not_found(index);
            return RestResp::json(error.status, &error.to_json_with_trace(false));
        }
        for shard in crate::shard_engines::search_shards(&state.metadata, index, params.get("routing").map(String::as_str)) {
            let Some(table) = state.routing_table.shard_table(index, shard) else {
                return unavailable_shard(index, shard);
            };
            let owners: BTreeSet<String> = table.searchable_copies(&state.nodes)
                .filter(|copy| copy.node_id.as_deref() != Some(node.local.id.as_str()) || local_holds(node, copy))
                .filter_map(|copy| copy.node_id.clone()).collect();
            let owner = if owners.contains(&node.local.id) { Some(node.local.id.clone()) } else { pick(&owners, request) };
            let Some(owner) = owner else { return unavailable_shard(index, shard) };
            groups.entry((owner, shard)).or_default().push(index.clone());
        }
    }
    let shard_count: usize = groups.values().map(Vec::len).sum();
    if shard_count == 1 {
        let ((owner, shard), indices) = groups.into_iter().next().unwrap();
        let mut child = request.clone();
        let prepared = (|| -> elasticsearch_engine::error::Result<()> {
            child.set_app_plan(Some(AppSearchPlan::try_json(&indices, query)?))?;
            child.set_shard(shard)?;
            child.set_read_copies(&indices)?;
            Ok(())
        })();
        if let Err(e)=prepared {return RestResp::json(e.status,&e.to_json())}
        return on_shard_copies(node, &child, candidates(&state, &child.app_plan.as_ref().unwrap().indices, shard, &owner), None).await.0;
    }
    if let Some(feature) = fanout::unsupported_feature(query, params) {
        return RestResp::error(500, "illegal_state_exception", &format!("{feature} cannot be combined across application shards yet"));
    }
    let (from, size) = fanout::from_size(query, params);
    let body = fanout::node_body(query, from, size);
    let futures = groups.into_iter().map(|((owner, shard), indices)| {
        let mut child = request.clone();
        let prepared = (|| -> elasticsearch_engine::error::Result<()> {
            child.set_uri(&crate::rest::util::strip_params(&child.uri, &["from", "size", "filter_path", "pretty"]))?;
            child.set_app_plan(Some(AppSearchPlan::try_json(&indices,&body)?))?;
            child.set_shard(shard)?;
            child.replace_header(crate::shard_engines::PREFILTER_SHARDS_HEADER,&shard_count.to_string())?;
            child.set_read_copies(&indices)?;
            Ok(())
        })();
        let count = indices.len();
        let copies = candidates(&state, &indices, shard, &owner);
        async move {
            if let Err(e)=prepared {return (RestResp::json(e.status,&e.to_json()),count)}
            let response = on_shard_copies(node, &child, copies, None).await.0;
            (response, count)
        }
    });
    let mut sources=elasticsearch_engine::source_json::ResponseSources::default();
    let mut parts = Vec::new();
    for (response, count) in futures::future::join_all(futures).await {
        if !response.is_success() { return response; }
        let Ok(mut part) = sources.parse(&response.body) else {
            return RestResp::error(500, "illegal_state_exception", "unreadable application shard response");
        };
        fanout::normalize_part_shards(&mut part, count);
        parts.push(part);
    }
    let merged = fanout::merge_search_with_limit(&parts, query, params, node.engine.max_buckets());
    if let Some(e) = fanout::take_agg_breaking() { return RestResp::json(e.status, &e.to_json()); }
    if let Some(e) = fanout::take_agg_error() { return RestResp::json(e.status, &e.to_json()); }
    let mut response=RestResp::json(200,&Value::Null);response.replace_body_vec(sources.render(&merged));response
}
