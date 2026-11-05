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
