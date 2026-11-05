//! `GET /{index}/_fleet/global_checkpoints` on a cluster: each shard's
//! global checkpoint is its primary's (a replica's engine copy does not
//! track it), so the coordinator asks every primary and waits for the
//! requested advance itself.
use super::*;

/// `(wait_for_advance, wait_for_index, checkpoints, timeout)` of the request
/// (already validated by the engine).
fn wait_params(params: &HashMap<String, String>) -> (bool, bool, Vec<i64>, Duration) {
    let flag = |k: &str| params.get(k).is_some_and(|v| v.is_empty() || v == "true");
    let checkpoints = params.get("checkpoints").map(|v| v.split(',').filter(|s| !s.is_empty()).filter_map(|s| s.trim().parse().ok()).collect()).unwrap_or_default();
    let timeout = params.get("timeout").and_then(|t| crate::rest::parse_timeout(t)).unwrap_or(Duration::from_secs(30));
    (flag("wait_for_advance"), flag("wait_for_index"), checkpoints, timeout)
}

/// Every shard's checkpoint above the wanted one?
fn advanced(current: &[i64], wanted: &[i64]) -> bool {
    current.iter().zip(wanted).all(|(c, w)| c > w)
}

pub(super) async fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> Option<RestResp> {
    let segs = req.segments();
    if segs.len() != 3 || segs[1] != "_fleet" || segs[2] != "global_checkpoints" || req.method() != Method::GET {
        return None;
    }
    // Validation, `wait_for_index` and errors exactly as the engine renders
    // them: a non-waiting run on this node's metadata.
    let probe_uri = crate::rest::util::strip_params(&req.uri, &["timeout"]);
    let probe_uri = format!("{probe_uri}{}timeout=0s", if probe_uri.contains('?') { "&" } else { "?" });
    let (wait_for_advance, wait_for_index, wanted, timeout) = wait_params(&req.params());
    let deadline = tokio::time::Instant::now() + timeout;
    let name = loop {
        let mut probe_req = req.clone();
        if let Err(error) = probe_req.set_uri(&probe_uri) { return Some(RestResp::json(error.status, &error.to_json())); }
        let probe = local_canonical(node, &probe_req).await;
        if probe.is_success() {
            break segs[0].clone();
        }
        if wait_for_index && probe.status == 404 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
            continue;
        }
        return Some(probe);
    };
    let engine = node.engine.clone();
    let index = match tokio::task::spawn_blocking(move || elasticsearch_engine::fleet::single_index(&engine, &name).map(|s| s.name.clone())).await {
        Ok(Ok(i)) => i,
        Ok(Err(e)) => return Some(RestResp::json(e.status, &e.to_json())),
        Err(e) => return Some(RestResp::error(500, "exception", &e.to_string())),
    };
    loop {
        let current = match primaries_checkpoints(node, req, &index).await {
            Ok(c) => c,
            Err(r) => return Some(r),
        };
        if !wait_for_advance || wanted.is_empty() {
            return Some(RestResp::json(200, &json!({"timed_out": false, "global_checkpoints": current})));
        }
        if wanted.len() != current.len() {
            return Some(RestResp::error(400, "status_exception", &format!("number of checkpoints must equal number of shards. [shard count: {}, checkpoint count: {}]", current.len(), wanted.len())));
        }
        if advanced(&current, &wanted) {
            return Some(RestResp::json(200, &json!({"timed_out": false, "global_checkpoints": current})));
        }
        if tokio::time::Instant::now() >= deadline {
            return Some(RestResp::json(200, &json!({"timed_out": true, "global_checkpoints": current})));
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Each shard's checkpoint as its primary reports it (`-1` without an
/// active primary).
async fn primaries_checkpoints(node: &Arc<ClusterNode>, req: &RestReq, index: &str) -> Result<Vec<i64>, RestResp> {
    let st = node.state();
    let n = st.metadata.indices.get(index).map(|m| m.number_of_shards.max(1)).unwrap_or(1);
    let mut out = Vec::with_capacity(n as usize);
    for k in 0..n {
        let Some(primary) = st.routing_table.shard_table(index, k).and_then(|t| t.active_primary()).and_then(|c| c.node_id.clone()) else {
            out.push(-1);
            continue;
        };
        let mut sub = req.clone();
        let prepare = (|| -> elasticsearch_engine::error::Result<()> {
            sub.set_method("GET")?;
            sub.set_uri(&format!("/{}/_fleet/global_checkpoints", encode_component(index)))?;
            sub.set_body(&[])?;
            sub.set_shard(k)?;
            Ok(())
        })();
        prepare.map_err(|e| RestResp::json(e.status, &e.to_json()))?;
        let r = if primary == node.local.id {
            local_canonical(node, &sub).await
        } else {
            sub.push_header(EXEC_LOCAL, "1").map_err(|e| RestResp::json(e.status, &e.to_json()))?;
            forward(node, &primary, &sub).await
        };
        if !r.is_success() {
            return Err(r);
        }
        let v = r.body_json().unwrap_or(Value::Null);
        out.push(v["global_checkpoints"][k as usize].as_i64().unwrap_or(-1));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_params_and_advance() {
        let p: HashMap<String, String> = [("wait_for_advance", "true"), ("checkpoints", "0,3"), ("timeout", "50ms")].into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        let (adv, idx, cps, t) = wait_params(&p);
        assert!(adv && !idx);
        assert_eq!(cps, vec![0, 3]);
        assert_eq!(t, Duration::from_millis(50));
        assert!(advanced(&[1, 4], &[0, 3]));
        assert!(!advanced(&[1, 3], &[0, 3]));
    }
}
