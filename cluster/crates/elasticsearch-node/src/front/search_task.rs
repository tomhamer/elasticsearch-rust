//! The coordinating search task (ES `SearchTask`): every client `_search`,
//! `_count`, `_knn_search`, `_search/template`, scroll page, `_msearch`
//! item and async search submission the cluster coordinates runs as a
//! cancellable task of this node, listed by `_tasks` with ES's action and
//! description.
//!
//! - Its id travels with every request derived from the search
//!   ([`PARENT_HEADER`](crate::rest::tasks::PARENT_HEADER), copied with the
//!   request): the engine running a shard request registers its shard task
//!   (`indices:data/read/search[phase/query]`, ...) as a child, listed with
//!   `parent_task_id`.
//! - In-process children follow the task's cancellation token directly
//!   ([`elasticsearch_http::SearchParent`]); children on other nodes are
//!   cancelled by the ban `_tasks/_cancel` broadcasts for the task (ES
//!   `TaskCancellationService` bans by parent id), and a shard request that
//!   arrives after the ban starts cancelled.
//! - Cancelled, the coordinator stops waiting for its outstanding shard
//!   requests and answers ES's `task_cancelled_exception`.
//! - When the client goes away before the answer (hyper drops the request's
//!   future when the connection closes, ES `RestCancellableNodeClient`),
//!   the task is cancelled and its children on other nodes banned.

use super::{err, RestReq, RestResp};
use crate::node::ClusterNode;
use crate::rest::tasks::PARENT_HEADER;
use elasticsearch_engine::rustic_core::cancel::Token;
use elasticsearch_engine::tasks::{Describe, Task};
use serde_json::{json, Value};
use std::sync::Arc;

pub(super) const SEARCH_ACTION: &str = "indices:data/read/search";
pub(super) const SCROLL_ACTION: &str = "indices:data/read/scroll";
pub(super) const ASYNC_SUBMIT_ACTION: &str = "indices:data/read/async_search/submit";

/// Why a task whose client went away was cancelled (ES: `http channel
/// [<channel>] closed`).
const CLOSED_REASON: &str = "http channel closed";

/// A coordinating search task whose shard requests this node's engines run
/// in-process: they follow its token without a registry lookup.
pub(super) struct Coordinating {
    id: String,
    token: Arc<Token>,
}

tokio::task_local! {
    static CURRENT: Arc<Coordinating>;
    /// The search task a synchronous cluster call's search runs under
    /// ([`linked`]); taken by the first search it coordinates.
    static LINKED: std::cell::RefCell<Option<(String, Option<Arc<Token>>)>>;
}

/// Run `work` (a synchronous caller's cluster call) so that the search it
/// coordinates runs as a child of task `parent` (id, token), cancelled
/// with it: a retriever's child searches are children of the client's
/// search task (ES), so a cancelled or disconnected client stops them.
pub(super) async fn linked<F: std::future::Future>(parent: Option<(String, Option<Arc<Token>>)>, work: F) -> F::Output {
    match parent {
        Some(parent) => LINKED.scope(std::cell::RefCell::new(Some(parent)), work).await,
        None => work.await,
    }
}

/// The parent [`linked`] set for the search being coordinated (once).
pub(super) fn take_linked() -> Option<(String, Option<Arc<Token>>)> {
    LINKED.try_with(|l| l.borrow_mut().take()).ok().flatten()
}

/// The registered task; finished when dropped, cancelled first (and its
/// children elsewhere banned) when dropped before the search answered.
struct Guard {
    node: Arc<ClusterNode>,
    task: Arc<Task>,
    id: String,
    done: bool,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if !self.done {
            self.task.cancel_with(CLOSED_REASON);
            ban_elsewhere(&self.node, &self.id);
        }
        self.node.engine.tasks.finish(&self.task, Ok(Value::Null));
    }
}

/// Ban `id`'s children on the other nodes (in the background): each
/// cancels the shard tasks it runs for it.
fn ban_elsewhere(node: &Arc<ClusterNode>, id: &str) {
    let others: Vec<String> = node.state().nodes.keys().filter(|n| n.as_str() != node.local.id.as_str()).cloned().collect();
    if !others.is_empty() {
        crate::rest::tasks::ban_on(node, others, id);
    }
}

/// Whether `req` is a client search this node coordinates as a task of
/// its own (not already part of another search: those carry the parent).
pub(super) fn coordinates(req: &RestReq) -> bool {
    if req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(PARENT_HEADER)) {
        return false;
    }
    let method = req.method.as_str();
    if method != "GET" && method != "POST" {
        return false;
    }
    let segs = req.segments();
    match segs.last().map(String::as_str) {
        Some("_search" | "_count" | "_knn_search") => true,
        // `_search/template` (not `_render/template`, `_msearch/template`).
        Some("template") => segs.len() >= 2 && segs[segs.len() - 2] == "_search",
        _ => false,
    }
}

/// A scroll page this node coordinates as a task of its own.
pub(super) fn coordinates_scroll(req: &RestReq) -> bool {
    req.method.as_str() != "DELETE" && !req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(PARENT_HEADER))
}

/// Whether `req` is a search shard request the engine runs as a child task.
pub(super) fn shard_route(req: &RestReq) -> bool {
    let segs = req.segments();
    match segs.last().map(String::as_str) {
        Some("_search" | "_count" | "_knn_search") => true,
        Some("scroll") => segs.len() >= 2 && segs[segs.len() - 2] == "_search" && req.method.as_str() != "DELETE",
        _ => false,
    }
}

/// The cancellation token of `parent` for a shard request this node's
/// engine runs: the coordinating task's own when this node coordinates it,
/// a cancelled one when `parent` was banned before the request arrived,
/// else none (a ban cancels the child task once registered).
pub(super) fn parent_token(node: &ClusterNode, parent: &str) -> Option<Arc<Token>> {
    if let Ok(Some(token)) = CURRENT.try_with(|c| (c.id == parent).then(|| c.token.clone())) {
        return Some(token);
    }
    if let Some((owner, n)) = parent.split_once(':') {
        if owner == node.local.id.as_str() {
            // Engine 0's task number is its cluster task id on this node.
            if let Some(task) = n.parse::<u64>().ok().and_then(|n| node.engine.tasks.get_running(n)) {
                return Some(task.cancel_token().clone());
            }
        }
    }
    if crate::rest::tasks::maybe_banned(parent) {
        return Some(Token::cancelled_with("by user request"));
    }
    None
}

/// ES `SearchRequest.buildDescription` of a coordinated search, built only
/// when a detailed listing asks for it.
fn describe(req: &RestReq, action: &str) -> Describe {
    let req = req.clone();
    let action = action.to_string();
    Arc::new(move || {
        let params = req.params();
        if action == SCROLL_ACTION {
            let id = req.json_ref().and_then(|b| b.get("scroll_id")).and_then(Value::as_str).map(str::to_string)
                .or_else(|| params.get("scroll_id").cloned()).unwrap_or_default();
            let keep = req.json_ref().and_then(|b| b.get("scroll")).and_then(Value::as_str).map(str::to_string)
                .or_else(|| params.get("scroll").cloned()).unwrap_or_default();
            return format!("scrollId[{id}], scroll[{keep}]");
        }
        let segs = req.segments();
        let indices = segs.first().filter(|s| segs.len() >= 2 && !s.starts_with('_')).cloned().unwrap_or_default();
        elasticsearch_http::search_task::search_description(&indices, &params, req.json_ref())
    })
}

/// ES `TaskCancelledException` (400): `task cancelled [<reason>]`.
pub(super) fn cancelled_response(token: &Token) -> RestResp {
    let reason = format!("task cancelled [{}]", token.reason().unwrap_or_else(|| "by user request".to_string()));
    RestResp::json(400, &json!({"error": {"root_cause": [{"type": "task_cancelled_exception", "reason": reason}], "type": "task_cancelled_exception", "reason": reason}, "status": 400}))
}

/// Run `work` as the search task `action` of `req`: a child of task
/// `parent_id` of this node when given, cancelled with `parent_token` when
/// given. `work` gets the request to run: `req` carrying the task as the
/// parent of everything derived from it. The answer is `work`'s, or ES's
/// `task_cancelled_exception` once the task is cancelled (outstanding shard
/// requests are then no longer waited for, and their tasks elsewhere are
/// banned).
pub(super) async fn run<F, Fut>(node: &Arc<ClusterNode>, req: &RestReq, action: &str, parent_id: Option<&str>, parent_token: Option<Arc<Token>>, work: F) -> RestResp
where
    F: FnOnce(RestReq) -> Fut,
    Fut: std::future::Future<Output = RestResp>,
{
    let opaque: Vec<(String, String)> = req.headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case("x-opaque-id")).map(|(_, v)| ("X-Opaque-Id".to_string(), v.clone())).collect();
    let task = match node.engine.tasks.try_register_linked(action, None, Some(describe(req, action)), true, false, &opaque, parent_id, parent_token) {
        Ok(task) => task,
        // Task metadata refused by the breaker: the search runs untracked
        // (as before tasks), not failed for its bookkeeping.
        Err(_) => return work(req.clone()).await,
    };
    // Engine 0's task number is its cluster task id on this node.
    let id = format!("{}:{}", node.local.id, task.id);
    let mut guard = Guard { node: node.clone(), task, id, done: false };
    let mut child = req.clone();
    if let Err(error) = child.replace_header(PARENT_HEADER, &guard.id) {
        // Nothing ran: the task just ends.
        guard.done = true;
        return err(error);
    }
    let token = guard.task.cancel_token().clone();
    let current = Arc::new(Coordinating { id: guard.id.clone(), token: token.clone() });
    let work = CURRENT.scope(current, work(child));
    tokio::pin!(work);
    let answer = tokio::select! {
        biased;
        answer = &mut work => Some(answer),
        _ = token.cancelled() => None,
    };
    guard.done = true;
    match answer {
        Some(answer) => answer,
        None => {
            // `_tasks/_cancel` of this task banned its children already; a
            // cancelled parent (an msearch, a deleted async search) did not.
            ban_elsewhere(node, &guard.id);
            cancelled_response(&token)
        }
    }
}

/// Cancellation of running cluster async searches by their stored key
/// (`DELETE _async_search/<id>` cancels the search, as ES deletes and
/// cancels the submit task).
static ASYNC: std::sync::LazyLock<parking_lot::Mutex<std::collections::HashMap<String, Arc<Token>>>> = std::sync::LazyLock::new(Default::default);

/// An async search `key` starts: the parent token of its submit task.
pub(super) fn async_started(key: &str) -> Arc<Token> {
    let token = Token::new();
    ASYNC.lock().insert(key.to_string(), token.clone());
    token
}

/// Async search `key` finished (its submit task ended).
pub(super) fn async_finished(key: &str) {
    ASYNC.lock().remove(key);
}

/// `DELETE _async_search/<key>`: cancel it if it still runs.
pub(super) fn async_cancel(key: &str) {
    if let Some(token) = ASYNC.lock().remove(key) {
        token.cancel("by user request");
    }
}
