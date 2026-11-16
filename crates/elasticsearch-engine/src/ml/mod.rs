//! Machine learning (`_ml`): anomaly-detection jobs, datafeeds, calendars,
//! filters, data frame analytics configs and `_evaluate`.
//!
//! Work package 7a covers configuration CRUD (with ES's validation and
//! messages), the job / datafeed state machine and the pure computations
//! (`_evaluate`, `_estimate_model_memory`, `_validate`). Nothing here runs an
//! anomaly detector or trains a model: an opened job and a started datafeed
//! hold their state but process no data.
//!
//! 7b adds the job process ([`process`]: data, buckets, results, model
//! snapshots, forecasts), the results APIs ([`results`]) over the results
//! index ([`resultsidx`]) and the datafeed runner ([`feed`]).

pub mod anomaly;
pub mod calendar;
pub mod dfa;
pub mod evaluate;
pub mod feed;
pub mod job;
pub mod datafeed;
pub mod memory;
pub mod mltok;
pub mod process;
pub mod results;
pub mod resultsidx;
pub mod preview;
pub mod store;
pub mod validate;
pub mod xc;

use crate::error::{EsError, Result};
use crate::node::{Node, ResolveOpts};
use serde_json::Value;

/// Run a search over an index expression (as the datafeed / DFA source
/// would); returns the search response JSON.
pub fn search(node: &Node, indices: &str, body: &Value) -> Result<Value> {
    // On a cluster the source shards are spread over the nodes.
    if let Some(c) = crate::coord::current() {
        return c("POST", &format!("/{}/_search?allow_no_indices=true", crate::coord::path_segment(indices)), Some(body.clone()));
    }
    let targets = node.resolve(indices, ResolveOpts { allow_no_indices: true, ..Default::default() })?;
    let req = crate::search::parse_request(body, &Default::default())?;
    let out = crate::search::execute(
        &targets,
        &req,
        &crate::search::ExecOpts { terms_lookup: None, cursors: None, max_buckets: node.max_buckets().max(65_536), scroll_id: None, pit_id: None },
    )?;
    serde_json::from_str(&out.json).map_err(|e| EsError::internal(e.to_string()))
}

/// Scroll keep-alive for ML extractors (ES `ScrollDataExtractor`: 5m).
const SCROLL_KEEP_ALIVE: &str = "5m";

/// A scroll over one search (ES `ScrollDataExtractor` / the analytics
/// reindex): every matching hit exactly once, in the request's sort order
/// (ties in hit order), over a view pinned when the scroll opened. A page
/// that fails leaves the scroll where it was, so retrying [`Scroll::next_page`]
/// neither skips nor repeats hits.
pub struct Scroll {
    indices: String,
    body: Value,
    state: ScrollState,
}

enum ScrollState {
    /// Not opened yet (opening is the first page).
    Fresh,
    /// Single node: the pinned shard views and per-shard cursors.
    Local { targets: Vec<crate::search::Target>, req: Box<crate::search::SearchRequest>, cursors: Vec<crate::search::Cursor> },
    /// Cluster: a REST scroll through the coordinator.
    Remote { coord: crate::coord::Coordinator, id: String },
    /// The position is gone (see [`Scroll::lost`]).
    Lost,
    Done,
}

impl Scroll {
    /// A scroll of `body` (with its own `size` and `sort`) over `indices`.
    pub fn new(indices: &str, body: Value) -> Scroll {
        Scroll { indices: indices.to_string(), body, state: ScrollState::Fresh }
    }

    /// The next page of hits; empty once the scroll is exhausted. On an
    /// error the scroll stays where it was (retry the page), unless
    /// [`Scroll::lost`]: the remote context moved past a partial page or
    /// expired, and the caller must reopen.
    pub fn next_page(&mut self, node: &Node) -> Result<Vec<Value>> {
        let resp = match &mut self.state {
            ScrollState::Done => return Ok(Vec::new()),
            ScrollState::Lost => return Err(EsError::search_context_missing("ml-extractor")),
            ScrollState::Fresh => match crate::coord::current() {
                Some(c) => {
                    let path = format!("/{}/_search?scroll={SCROLL_KEEP_ALIVE}&allow_no_indices=true", crate::coord::path_segment(&self.indices));
                    let resp = c("POST", &path, Some(self.body.clone()))?;
                    let id = resp["_scroll_id"].as_str().map(str::to_string);
                    if let Err(e) = page_failure(&resp) {
                        if let Some(id) = id {
                            let _ = c("DELETE", "/_search/scroll", Some(serde_json::json!({"scroll_id": id})));
                        }
                        return Err(e);
                    }
                    self.state = match id {
                        Some(id) => ScrollState::Remote { coord: c, id },
                        None => ScrollState::Done,
                    };
                    resp
                }
                None => {
                    let targets = node.resolve(&self.indices, ResolveOpts { allow_no_indices: true, ..Default::default() })?;
                    let req = crate::search::parse_request(&self.body, &Default::default())?;
                    let targets = node.snapshot(&targets)?;
                    let out = crate::search::execute(&targets, &req, &Self::opts(node, None))?;
                    let resp: Value = serde_json::from_str(&out.json).map_err(|e| EsError::internal(e.to_string()))?;
                    page_failure(&resp)?;
                    self.state = ScrollState::Local { targets, req: Box::new(req), cursors: out.cursors };
                    resp
                }
            },
            ScrollState::Local { targets, req, cursors } => {
                let out = crate::search::execute(targets, req, &Self::opts(node, Some(cursors.as_slice())))?;
                let resp: Value = serde_json::from_str(&out.json).map_err(|e| EsError::internal(e.to_string()))?;
                page_failure(&resp)?;
                // Cursors move only once the page succeeded.
                *cursors = out.cursors;
                resp
            }
            ScrollState::Remote { coord, id } => {
                let resp = match coord("POST", "/_search/scroll", Some(serde_json::json!({"scroll": SCROLL_KEEP_ALIVE, "scroll_id": id.clone()}))) {
                    Ok(r) => r,
                    Err(e) => {
                        if e.status == 404 || e.kind.contains("search_context_missing") {
                            self.close_lost();
                        }
                        return Err(e);
                    }
                };
                if let Some(n) = resp["_scroll_id"].as_str() {
                    *id = n.to_string();
                }
                if let Err(e) = page_failure(&resp) {
                    // The server-side cursors already moved past this page.
                    self.close_lost();
                    return Err(e);
                }
                resp
            }
        };
        let hits = resp["hits"]["hits"].as_array().cloned().unwrap_or_default();
        if hits.is_empty() {
            self.close();
        }
        Ok(hits)
    }

    /// Whether the scroll lost its position (reopen to continue).
    pub fn lost(&self) -> bool {
        matches!(self.state, ScrollState::Lost)
    }

    /// Tests: the scroll's position is lost (as an expired remote context).
    #[cfg(test)]
    pub(crate) fn lose(&mut self) {
        self.close_lost();
    }

    fn close_lost(&mut self) {
        self.close();
        self.state = ScrollState::Lost;
    }

    fn opts<'a>(node: &Node, cursors: Option<&'a [crate::search::Cursor]>) -> crate::search::ExecOpts<'a> {
        crate::search::ExecOpts { terms_lookup: None, cursors, max_buckets: node.max_buckets().max(65_536), scroll_id: None, pit_id: None }
    }

    /// Release the pinned view / the remote scroll context.
    pub fn close(&mut self) {
        if let ScrollState::Remote { coord, id } = std::mem::replace(&mut self.state, ScrollState::Done) {
            let _ = coord("DELETE", "/_search/scroll", Some(serde_json::json!({"scroll_id": id})));
        }
    }
}

/// A page with shard failures would silently drop the failed shards' hits.
/// A search response every shard answered in full: shard failures, or a
/// timed out search, are an error (the extractor retries the page).
pub(crate) fn require_complete(resp: &Value) -> Result<()> {
    page_failure(resp)?;
    if resp["_shards"]["failed"].as_u64().unwrap_or(0) > 0 {
        return Err(EsError::new(503, "search_phase_execution_exception", "Partial shard failure"));
    }
    if resp["timed_out"].as_bool() == Some(true) {
        return Err(EsError::new(503, "search_phase_execution_exception", "The search timed out before every shard answered"));
    }
    Ok(())
}

fn page_failure(resp: &Value) -> Result<()> {
    match resp["_shards"]["failures"].as_array().and_then(|a| a.first()) {
        Some(f) => {
            let reason = f["reason"]["reason"].as_str().unwrap_or("shard failure");
            Err(EsError::new(503, "search_phase_execution_exception", format!("Partial shard failure: {reason}")))
        }
        None => Ok(()),
    }
}

impl Drop for Scroll {
    fn drop(&mut self) {
        self.close();
    }
}

/// Is ML upgrade mode on (`POST _ml/set_upgrade_mode?enabled=true`)?
pub fn upgrade_mode(node: &Node) -> bool {
    store::service(node).upgrade_mode()
}

/// Ids matching a comma-separated expression of ids / wildcards / `_all`.
pub fn matches_expr(expr: &str, id: &str) -> bool {
    expr.split(',').map(str::trim).any(|p| p == "_all" || p == "*" || crate::mapping::wildcard_match(p, id))
}

/// Does the expression contain a wildcard or `_all`?
pub fn is_wildcard_expr(expr: &str) -> bool {
    expr.split(',').map(str::trim).any(|p| p == "_all" || p.contains('*'))
}

/// ES `MlStrings.isValidId`: lowercase alphanumerics, `-`, `_`, `.`;
/// starts and ends alphanumeric.
pub fn is_valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_' || b == b'.')
        && id.as_bytes()[0].is_ascii_alphanumeric()
        && id.as_bytes()[id.len() - 1].is_ascii_alphanumeric()
}

/// ES `MlStrings.ID_LENGTH_LIMIT`.
pub const ID_LENGTH_LIMIT: usize = 64;
