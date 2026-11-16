//! SQL on a cluster.
//!
//! SQL runs on the ES|QL executor, so a query runs like ES|QL
//! (`esql_on_copies`, remote shard reads through `esql_dist`). Cursors and
//! stored async results live on the node that ran the query, so follow-ups
//! go to the master first and then to the other nodes.

use super::{esql_on_copies, forward, local_canonical, RestReq, RestResp, EXEC_LOCAL};
use crate::node::ClusterNode;
use serde_json::Value;
use std::sync::Arc;

/// What a `/_sql...` request is, for routing.
#[derive(Debug, PartialEq)]
pub(super) enum Kind {
    /// A new SQL query: like ES|QL.
    Sql,
    /// A request naming state kept by the node that ran a query (cursor
    /// page, cursor close, async result/status/delete).
    FollowUp,
}

pub(super) fn kind(method: &str, segments: &[String], body: &[u8]) -> Option<Kind> {
    if segments.first().map(String::as_str) != Some("_sql") {
        return None;
    }
    match (method, segments.get(1).map(String::as_str)) {
        ("GET" | "POST", None) => {
            let cursor = serde_json::from_slice::<Value>(body).ok()
                .and_then(|b| b.get("cursor").and_then(Value::as_str).map(|c| !c.is_empty()))
                .unwrap_or(false);
            Some(if cursor { Kind::FollowUp } else { Kind::Sql })
        }
        ("POST", Some("close")) => Some(Kind::FollowUp),
        (_, Some("async")) => Some(Kind::FollowUp),
        _ => None,
    }
}

/// Whether a follow-up's answer says this node does not hold its state.
fn state_missing(resp: &RestResp, closing: bool) -> bool {
    let Some(v) = resp.body_json() else { return false };
    let reason = v["error"]["reason"].as_str().unwrap_or("");
    (closing && resp.status == 200 && v["succeeded"] == false)
        || (resp.status == 400 && reason.starts_with("Unknown cursor"))
        || (resp.status == 404 && (v["error"]["type"] == "resource_not_found_exception"
            || (v["error"]["type"] == "search_phase_execution_exception"
                && v["error"]["caused_by"]["type"] == "search_context_missing_exception")))
}

/// Whether a forwarded follow-up never reached a handler (the peer was
/// unreachable): that peer is not known to hold the state, so the search
/// for its owner goes on (review finding 613).
fn peer_unreachable(resp: &RestResp) -> bool {
    let Some(v) = resp.body_json() else { return false };
    matches!(
        v["error"]["type"].as_str().unwrap_or(""),
        "node_not_connected_exception" | "connect_transport_exception" | "node_disconnected_exception" | "node_closed_exception"
            | "receive_timeout_transport_exception" | "send_request_transport_exception" | "transport_exception"
    )
}

/// A SQL follow-up's owner search over the nodes in `order` (the master
/// first): `ask(id)` runs it on node `id` (`local`: this node). The first
/// answer from a node that holds the state is the answer; a node that does
/// not, or that cannot be reached (it is not known to own it), lets the
/// search go on (review finding 613). With no owner found: an unreachable
/// node's failure (it may hold it), else the missing-state answer.
async fn follow_up<F, Fut>(order: Vec<String>, local: &str, closing: bool, ask: F) -> RestResp
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = RestResp>,
{
    let mut first: Option<RestResp> = None;
    let mut unreachable: Option<RestResp> = None;
    for id in order {
        let remote = id != local;
        let resp = ask(id).await;
        if remote && peer_unreachable(&resp) {
            unreachable.get_or_insert(resp);
            continue;
        }
        if !state_missing(&resp, closing) {
            return resp;
        }
        first.get_or_insert(resp);
    }
    unreachable.or(first).unwrap_or_else(|| RestResp::error(503, "master_not_discovered_exception", ""))
}

pub(super) async fn handle(node: &Arc<ClusterNode>, req: &RestReq, kind: Kind) -> RestResp {
    match kind {
        Kind::Sql => esql_on_copies(node, req).await,
        Kind::FollowUp => {
            let closing = req.segments().get(1).is_some_and(|s| s == "close");
            let state = node.state();
            let master = state.master_node_id.clone();
            let mut order: Vec<String> = master.iter().cloned().collect();
            order.extend(state.nodes.keys().filter(|id| Some(*id) != master.as_ref()).cloned());
            let local = node.local.id.clone();
            follow_up(order, &local, closing, |id| async move {
                if id == node.local.id {
                    local_canonical(node, req).await
                } else {
                    let mut sub = req.clone();
                    if let Err(error) = sub.push_header(EXEC_LOCAL, "1") { return RestResp::json(error.status, &error.to_json()); }
                    forward(node, &id, &sub).await
                }
            }).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs(p: &str) -> Vec<String> {
        p.split('/').filter(|s| !s.is_empty()).map(str::to_string).collect()
    }

    #[test]
    fn sql_missing_cursor_routes_to_its_owner_but_other_errors_do_not() {
        use serde_json::json;
        let missing = RestResp::json(404, &json!({"error":{
            "type":"search_phase_execution_exception",
            "caused_by":{"type":"search_context_missing_exception"}
        }}));
        assert!(state_missing(&missing, false));
        assert!(!state_missing(&RestResp::json(404, &json!({"error":{
            "type":"search_phase_execution_exception",
            "caused_by":{"type":"index_not_found_exception"}
        }})), false));
        assert!(state_missing(&RestResp::json(200, &json!({"succeeded":false})), true));
        assert!(!state_missing(&RestResp::json(200, &json!({"succeeded":false})), false));
        assert!(!state_missing(&RestResp::json(200, &json!({"succeeded":true})), true));
        assert!(!state_missing(&RestResp::json(403, &json!({"error":{"type":"security_exception"}})), false));
    }

    #[test]
    fn sql_queries_run_on_copies_and_follow_ups_find_their_node() {
        assert_eq!(kind("POST", &segs("/_sql"), br#"{"query":"SELECT 1"}"#), Some(Kind::Sql));
        assert_eq!(kind("GET", &segs("/_sql"), b""), Some(Kind::Sql));
        assert_eq!(kind("POST", &segs("/_sql"), br#"{"cursor":"abc"}"#), Some(Kind::FollowUp));
        assert_eq!(kind("POST", &segs("/_sql/close"), br#"{"cursor":"abc"}"#), Some(Kind::FollowUp));
        assert_eq!(kind("GET", &segs("/_sql/async/xyz"), b""), Some(Kind::FollowUp));
        assert_eq!(kind("DELETE", &segs("/_sql/async/delete/xyz"), b""), Some(Kind::FollowUp));
        // Translation reads mappings only; stats are per node.
        assert_eq!(kind("POST", &segs("/_sql/translate"), b"{}"), None);
        assert_eq!(kind("GET", &segs("/_sql/stats"), b""), None);
        // EQL runs on the master (`classify`) and reads through the coordinator.
        assert_eq!(kind("POST", &segs("/logs/_eql/search"), b"{}"), None);
        assert_eq!(kind("POST", &segs("/_search"), b"{}"), None);
    }

    /// Review finding 613: a peer that cannot be reached is not the
    /// cursor's owner; the owner's own errors still end the search.
    #[test]
    fn unreachable_peers_do_not_end_the_owner_search() {
        use serde_json::json;
        let down = RestResp::error(502, "node_not_connected_exception", "[n1] node not connected");
        assert!(peer_unreachable(&down));
        let timeout = RestResp::error(500, "receive_timeout_transport_exception", "timed out");
        assert!(peer_unreachable(&timeout));
        let owner_error = RestResp::json(400, &json!({"error": {"type": "illegal_argument_exception", "reason": "bad"}, "status": 400}));
        assert!(!peer_unreachable(&owner_error));
        assert!(!state_missing(&owner_error, false));
    }

    /// Review finding 613 across nodes: the master holds no cursor, an
    /// earlier peer is disconnected, the owner answers its page. The
    /// follow-up reaches the owner (each node asked once, in order); with
    /// the owner itself down the peer's failure is the answer; with no
    /// owner anywhere, the missing cursor.
    #[tokio::test]
    async fn follow_ups_reach_their_owner_past_an_unreachable_peer() {
        use serde_json::json;
        let missing = || RestResp::json(404, &json!({"error": {"type": "search_phase_execution_exception", "reason": "all shards failed", "caused_by": {"type": "search_context_missing_exception", "reason": "No search context found"}}, "status": 404}));
        let down = || RestResp::error(502, "node_not_connected_exception", "[b] node not connected");
        let page = || RestResp::json(200, &json!({"rows": [[1]], "cursor": "next"}));
        let order = || vec!["m".to_string(), "b".to_string(), "c".to_string()];
        let asked = std::sync::Mutex::new(Vec::new());
        let r = follow_up(order(), "m", false, |id| {
            asked.lock().unwrap().push(id.clone());
            let resp = match id.as_str() { "m" => missing(), "b" => down(), _ => page() };
            async move { resp }
        }).await;
        assert_eq!(r.status, 200);
        assert_eq!(*asked.lock().unwrap(), vec!["m", "b", "c"]);
        // The owner down too: the unreachable failure, not "missing".
        let r = follow_up(order(), "m", false, |id| { let resp = match id.as_str() { "m" => missing(), _ => down() }; async move { resp } }).await;
        assert_eq!(r.status, 502);
        // Nobody has it.
        let r = follow_up(order(), "m", false, |_| async { missing() }).await;
        assert_eq!(r.status, 404);
        // An owner's own error ends the search.
        let r = follow_up(order(), "m", false, |id| {
            let resp = if id == "m" { RestResp::json(400, &json!({"error": {"type": "illegal_argument_exception", "reason": "bad page"}, "status": 400})) } else { page() };
            async move { resp }
        }).await;
        assert_eq!(r.status, 400);
    }
}
