//! Search tasks of the engine's search handlers (ES `SearchTask` /
//! `SearchShardTask`): every `_search`, `_count`, `_knn_search` and scroll
//! page runs as a cancellable task listed by `_tasks`.
//!
//! - Run for a cluster coordinator ([`crate::SearchParent`]): a shard-level
//!   child, `indices:data/read/search[phase/...]`, whose `parent_task_id` is
//!   the coordinator's task and whose cancellation follows it (in-process
//!   through the parent's token; on another node through the parent's ban,
//!   which cancels the children registered for it).
//! - Received directly (the engine's own HTTP server): the coordinating
//!   `indices:data/read/search` (`indices:data/read/scroll`) task itself.
//! - Other internal searches (a feature's own sub-search) run untracked.
//!
//! The work runs under the task's token ([`elasticsearch_engine::rustic_core::cancel::search_scope`])
//! on the executor thread: the search loops poll it. A handler whose future
//! is dropped before it answers (the client went away, or the coordinator
//! stopped waiting) cancels its task, so the executor thread stops at its
//! next check instead of finishing work nobody reads.

use crate::Req;
use elasticsearch_engine::error::Result;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::rustic_core::cancel::Token;
use elasticsearch_engine::tasks::Task;
use serde_json::Value;
use std::sync::Arc;

pub(crate) const SEARCH_ACTION: &str = "indices:data/read/search";
pub(crate) const SCROLL_ACTION: &str = "indices:data/read/scroll";
const QUERY_PHASE: &str = "indices:data/read/search[phase/query]";
const FETCH_PHASE: &str = "indices:data/read/search[phase/fetch/id]";
const DFS_PHASE: &str = "indices:data/read/search[phase/dfs]";
const SCROLL_PHASE: &str = "indices:data/read/search[phase/query/scroll]";

/// Why a dropped handler's task was cancelled (ES `RestCancellableNodeClient`
/// cancels with `http channel [<channel>] closed`).
pub(crate) const CLOSED_REASON: &str = "http channel closed";

/// The route a search task runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// `_search`, `_count`, `_knn_search`.
    Search,
    /// A scroll page (`_search/scroll`).
    Scroll,
}

/// A running search task, held by the work it runs: finished when dropped,
/// and cancelled first when dropped before [`Guard::complete`] (its work
/// never ran or panicked).
pub(crate) struct Guard {
    node: Arc<Node>,
    task: Arc<Task>,
    done: bool,
}

impl Guard {
    /// The token the search work runs under.
    pub(crate) fn token(&self) -> Arc<Token> {
        self.task.cancel_token().clone()
    }

    /// The handler answered: its task ends without being cancelled.
    pub(crate) fn complete(&mut self) {
        self.done = true;
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if !self.done {
            self.task.cancel_with(CLOSED_REASON);
        }
        self.node.tasks.finish(&self.task, Ok(Value::Null));
    }
}

/// Cancels a search's token when dropped armed: the handler awaiting the
/// search's work was dropped (its client or coordinator went away).
pub(crate) struct CancelOnDrop(Option<Arc<Token>>);

impl CancelOnDrop {
    pub(crate) fn new(token: Arc<Token>) -> Self {
        CancelOnDrop(Some(token))
    }
    /// The work answered.
    pub(crate) fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(token) = self.0.take() {
            token.cancel(CLOSED_REASON);
        }
    }
}

fn opaque(req: &Req) -> Vec<(String, String)> {
    req.headers.get("x-opaque-id").and_then(|v| v.to_str().ok()).map(|v| vec![("X-Opaque-Id".to_string(), v.to_string())]).unwrap_or_default()
}

/// The shard-level action of a coordinator's shard request.
fn phase_action(req: &Req, kind: Kind) -> &'static str {
    if req.extensions.get::<crate::DfsPhaseOnly>().is_some() {
        return DFS_PHASE;
    }
    if matches!(req.extensions.get::<crate::InternalQtf>(), Some(crate::InternalQtf::Fetch | crate::InternalQtf::Rerun)) {
        return FETCH_PHASE;
    }
    if kind == Kind::Scroll { SCROLL_PHASE } else { QUERY_PHASE }
}

/// ES `SearchRequest.buildDescription`: `indices[a,b], search_type[QUERY_THEN_FETCH], scroll[1m], source[{...}]`
/// (the source as the request gave it, compact).
pub fn search_description(indices: &str, params: &std::collections::HashMap<String, String>, body: Option<&Value>) -> String {
    let search_type = match params.get("search_type").map(String::as_str) {
        Some("dfs_query_then_fetch") => "DFS_QUERY_THEN_FETCH",
        _ => "QUERY_THEN_FETCH",
    };
    let scroll = params.get("scroll").map(|s| format!("scroll[{s}], ")).unwrap_or_default();
    let source = body.filter(|b| b.as_object().is_some_and(|o| !o.is_empty())).map(|b| b.to_string()).unwrap_or_default();
    format!("indices[{indices}], search_type[{search_type}], {scroll}source[{source}]")
}

/// Register the task `req` runs as (see the module docs); `None` for an
/// untracked internal search.
pub(crate) fn begin(node: &Arc<Node>, req: &Req, kind: Kind, index: Option<&str>) -> Result<Option<Guard>> {
    let headers = opaque(req);
    let task = if let Some(parent) = req.extensions.get::<crate::SearchParent>() {
        // ES `ShardSearchRequest.getDescription`: `shardId[[index][n]]`.
        let shard = crate::READ_SHARD.try_with(|s| (*s).map(|s| s.0)).ok().flatten().unwrap_or(0);
        let description = format!("shardId[[{}][{shard}]]", index.unwrap_or_default());
        node.tasks.try_register_linked(phase_action(req, kind), Some(description.as_str()), None, true, false, &headers, Some(parent.id.as_str()), parent.token.clone())?
    } else if req.extensions.get::<crate::InternalHttpDispatch>().is_none() {
        let (action, describe) = match kind {
            Kind::Search => {
                let indices = index.unwrap_or_default().to_string();
                let params = req.params.clone();
                let body = req.body.clone();
                let describe: elasticsearch_engine::tasks::Describe = Arc::new(move || {
                    let parsed = serde_json::from_slice::<Value>(&body).ok();
                    search_description(&indices, &params, parsed.as_ref())
                });
                (SEARCH_ACTION, describe)
            }
            Kind::Scroll => {
                let params = req.params.clone();
                let describe: elasticsearch_engine::tasks::Describe = Arc::new(move || {
                    format!("scrollId[{}], scroll[{}]", params.get("scroll_id").cloned().unwrap_or_default(), params.get("scroll").cloned().unwrap_or_default())
                });
                (SCROLL_ACTION, describe)
            }
        };
        node.tasks.try_register_linked(action, None, Some(describe), true, false, &headers, None, None)?
    } else {
        return Ok(None);
    };
    // A parent banned before this shard request arrived hands a cancelled
    // token: the task starts cancelled and stops at its first check.
    Ok(Some(Guard { node: node.clone(), task, done: false }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn describes_a_search_as_es_does() {
        let mut params = std::collections::HashMap::new();
        assert_eq!(search_description("test", &params, Some(&json!({"query": {"match_all": {}}}))),
            r#"indices[test], search_type[QUERY_THEN_FETCH], source[{"query":{"match_all":{}}}]"#);
        params.insert("search_type".to_string(), "dfs_query_then_fetch".to_string());
        params.insert("scroll".to_string(), "1m".to_string());
        assert_eq!(search_description("a,b", &params, None), "indices[a,b], search_type[DFS_QUERY_THEN_FETCH], scroll[1m], source[]");
    }
}
