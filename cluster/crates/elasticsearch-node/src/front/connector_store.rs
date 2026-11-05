//! Connector documents belong to their system index's primary, not the master.
use super::*;

pub(super) async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> Option<RestResp> {
    let parts: Vec<_> = req.path().trim_start_matches('/').split('/').collect();
    if parts.first() != Some(&"_connector") || parts.get(1).is_some_and(|p| (p.starts_with('_') && !matches!(*p, "_secret" | "_sync_job")) || p.is_empty()) {
        return None;
    }
    if req.method() == Method::DELETE && parts.len() == 2 && !parts[1].starts_with('_')
        && req.params().get("delete_sync_jobs").is_some_and(|v| v == "true") {
        return Some(delete_with_jobs(node, req, &decode_path(parts[1])).await);
    }
    let secret = parts.get(1) == Some(&"_secret");
    let sync_job = parts.get(1) == Some(&"_sync_job");
    let index = if secret { elasticsearch_engine::connector::SECRETS_INDEX } else if sync_job { elasticsearch_engine::connector::SYNC_JOB_INDEX } else { elasticsearch_engine::connector::CONNECTOR_INDEX };
    let creates = if sync_job { req.method() == Method::POST && parts.len() == 2 }
        else { matches!(req.method(), Method::PUT | Method::POST) && parts.len() <= if secret { 3 } else { 2 } };
    if !node.state().metadata.indices.contains_key(index) {
        if !creates { return Some(local_canonical(node, req).await); }
        if !node.is_master() { return Some(crate::rest::forward_to_master(node, req).await); }
        let created = node.mutate_engine_metadata(move |engine| {
            if engine.get_index(index).is_some() { return Ok(()); }
            engine.create_index(index, &json!({"settings": {"index": {
                "number_of_shards": 1, "auto_expand_replicas": "0-1", "hidden": true
            }}, "mappings": {"dynamic": false}})).map(|_| ())
        }).await;
        match created {
            Ok(((), version)) => { node.wait_applied_everywhere(version, request_timeout(req)).await; }
            Err(error) => return Some(RestResp::json(error.status, &error.to_json_with_trace(false))),
        }
    }
    let deadline = tokio::time::Instant::now() + request_timeout(req);
    loop {
        if let Some(owner) = primary_node(node, index) {
            if owner != node.local.id { return Some(forward(node, &owner, req).await); }
            if let Some(primary) = node.shards.primary(index, 0) {
                let response = if req.method() == Method::GET || req.method() == Method::HEAD {
                    let _guard = primary.gate.read().await;
                    if !still_primary(node, &primary) || !engine_is(&node.engine, &primary) {
                        no_longer_primary(&primary)
                    } else { local_canonical(node, req).await }
                } else { execute_write(node, &primary, req).await };
                return Some(response);
            }
        }
        if !wait_state_change(node, deadline).await { return Some(unavailable(index)); }
    }
}

/// Each store acknowledges its own replicated operation. Do not acknowledge
/// the connector request if any associated job deletion fails.
async fn delete_with_jobs(node: &Arc<ClusterNode>, req: &RestReq, id: &str) -> RestResp {
    let mut deletion = req.clone();
    let mut params: Vec<_> = req.query().split('&').filter(|p| !p.is_empty() && decode(p.split('=').next().unwrap_or("")) != "delete_sync_jobs").collect();
    params.push("delete_sync_jobs=false");
    if let Err(error) = deletion.set_uri(&(format!("{}?{}", req.path(), params.join("&")))) { return RestResp::json(error.status, &error.to_json()); }
    let result = Box::pin(dispatch(node, deletion)).await;
    if !result.is_success() { return result; }
    let deadline = tokio::time::Instant::now() + request_timeout(req);
    loop {
        let mut list = req.clone();
        if let Err(error) = list.set_method("GET") { return RestResp::json(error.status, &error.to_json()); }
        if let Err(error) = list.set_uri(&(format!("/_connector/_sync_job?connector_id={}&size=1000", encode_component(id)))) { return RestResp::json(error.status, &error.to_json()); }
        if let Err(error) = list.set_body(&[]) { return RestResp::json(error.status, &error.to_json()); }
        let response = Box::pin(dispatch(node, list)).await;
        if !response.is_success() { return response; }
        let value = response.body_json().unwrap_or(Value::Null);
        let Some(jobs) = value["results"].as_array() else {
            return RestResp::error(500, "exception", "invalid connector sync job list response");
        };
        if jobs.is_empty() { return result; }
        for job in jobs {
            let Some(job_id) = job["id"].as_str() else { return RestResp::error(500, "exception", "sync job has no identifier"); };
            let mut deletion = req.clone();
            if let Err(error) = deletion.set_uri(&(format!("/_connector/_sync_job/{}", encode_component(job_id)))) { return RestResp::json(error.status, &error.to_json()); }
            if let Err(error) = deletion.set_body(&[]) { return RestResp::json(error.status, &error.to_json()); }
            let response = Box::pin(dispatch(node, deletion)).await;
            if !response.is_success() && response.status != 404 { return response; }
        }
        if tokio::time::Instant::now() >= deadline {
            return RestResp::error(503, "timeout_exception", "timed out deleting connector sync jobs");
        }
    }
}
