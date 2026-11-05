//! Fleet support (`x-pack/plugin/fleet`): per-shard global checkpoints,
//! the `wait_for_checkpoints` pre-check of fleet searches, and the fleet
//! secrets store (`.fleet-secrets-7`, aliased `.fleet-secrets`).
//!
//! A standalone node has no replicas, so a shard's global checkpoint is its
//! local checkpoint. Shards of a cluster-managed (sparse) index report the
//! checkpoint replication maintains.

use crate::error::{EsError, Result};
use crate::index::{IndexService, IndexShard, WriteReq};
use crate::node::{Node, ResolveOpts};
use crate::shard::WriteCond;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The global checkpoint of one shard copy.
pub fn shard_checkpoint(svc: &IndexService, shard: &IndexShard) -> i64 {
    if svc.is_sparse() {
        shard.global_checkpoint()
    } else {
        shard.shard.local_checkpoint().max(shard.global_checkpoint())
    }
}

/// Highest seq_no issued on a shard (`-1` when nothing was written).
pub fn max_issued(shard: &IndexShard) -> i64 {
    shard.shard.max_seq_no()
}

/// Global checkpoints of every shard, in shard order (`-1` for a shard
/// with no local copy).
pub fn global_checkpoints(svc: &IndexService) -> Vec<i64> {
    (0..svc.number_of_shards()).map(|n| svc.shard(n).map(|s| shard_checkpoint(svc, &s)).unwrap_or(-1)).collect()
}

/// `no such index [name]`, as ES reports a data stream name given to an
/// API that only takes indices and aliases.
fn ds_excluded(name: &str) -> EsError {
    EsError::new(404, "index_not_found_exception", format!("no such index [{name}]"))
        .with("resource.type", "index_or_alias")
        .with("excluded_ds", "true")
        .with("resource.id", name)
        .with("index_uuid", "_na_")
        .with("index", name)
}

/// The single index `name` (an index or an alias of exactly one index)
/// names; wildcards and lists are taken literally.
pub fn single_index(node: &Node, name: &str) -> Result<Arc<IndexService>> {
    if let Some(s) = node.get_index(name) {
        return Ok(s);
    }
    if name.contains('*') || name.contains(',') || name.is_empty() {
        return Err(EsError::index_not_found(name));
    }
    if node.data_stream(name).is_some() {
        return Err(ds_excluded(name));
    }
    let opts = ResolveOpts { allow_no_indices: false, ..ResolveOpts::default() };
    let targets = node.resolve(name, opts).map_err(|_| EsError::index_not_found(name))?;
    match targets.as_slice() {
        [t] => Ok(t.svc.clone()),
        [] => Err(EsError::index_not_found(name)),
        _ => Err(EsError::illegal_argument(format!(
            "alias [{name}] has more than one index associated with it [{}], can't execute a single index op",
            targets.iter().map(|t| t.svc.name.clone()).collect::<Vec<_>>().join(", ")
        ))),
    }
}

/// `GET /{index}/_fleet/global_checkpoints`: `(timed_out, checkpoints)`.
/// With `wait_for_advance`, waits (up to `timeout`) until every shard's
/// checkpoint is above the given one; with `wait_for_index`, first waits
/// for the index to exist.
pub fn get_global_checkpoints(node: &Node, index: &str, wait_for_advance: bool, wait_for_index: bool, checkpoints: &[i64], timeout: Duration) -> Result<(bool, Vec<i64>)> {
    let deadline = Instant::now() + timeout;
    let svc = loop {
        match single_index(node, index) {
            Ok(s) => break s,
            Err(e) if wait_for_index && e.status == 404 && Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
            Err(e) => return Err(e),
        }
    };
    let mut cur = global_checkpoints(&svc);
    if !wait_for_advance || checkpoints.is_empty() {
        return Ok((false, cur));
    }
    if checkpoints.len() != cur.len() {
        return Err(EsError::new(400, "status_exception", format!(
            "number of checkpoints must equal number of shards. [shard count: {}, checkpoint count: {}]",
            cur.len(),
            checkpoints.len()
        )));
    }
    loop {
        if cur.iter().zip(checkpoints).all(|(c, want)| c > want) {
            return Ok((false, cur));
        }
        if Instant::now() >= deadline {
            return Ok((true, cur));
        }
        std::thread::sleep(Duration::from_millis(5));
        cur = global_checkpoints(&svc);
    }
}

/// Why `wait_for_checkpoints` fails on one shard (an `illegal_argument_exception`
/// reported as a shard failure), or `None` when the shard can be searched.
/// Writes are visible as soon as they are applied here, so a checkpoint at or
/// below the highest issued seq_no needs no wait for a refresh.
pub fn checkpoint_failure(svc: &IndexService, shard: Option<&IndexShard>, checkpoint: i64) -> Option<String> {
    if svc.refresh_interval_ms() < 0 {
        return Some("Cannot use wait_for_checkpoints with [index.refresh_interval=-1]".into());
    }
    if checkpoint < 0 {
        return None;
    }
    let issued = shard.map(max_issued).unwrap_or(-1);
    if checkpoint > issued {
        return Some(format!("Cannot wait for unissued seqNo checkpoint [wait_for_checkpoint={checkpoint}, max_issued_seqNo={issued}]"));
    }
    None
}

/// The longest a fleet search waits for its checkpoints, whatever
/// `wait_for_checkpoints_timeout` says (a huge value never overflows the
/// deadline into no wait at all).
pub const MAX_WAIT: Duration = Duration::from_secs(300);

/// Whether a copy has caught up for a fleet search's `checkpoint` (ES
/// `SearchService`'s wait): every operation up to it is visible to searches
/// (refreshed), and the shard's global checkpoint has reached it, so a copy
/// never returns documents not yet acknowledged.
pub fn caught_up(svc: &IndexService, shard: &IndexShard, checkpoint: i64) -> bool {
    checkpoint < 0 || (shard.shard.refreshed_checkpoint() >= checkpoint && shard_checkpoint(svc, shard) >= checkpoint)
}

/// ES's wait of a fleet search for `wait_for_checkpoints` on one shard
/// copy, at most `timeout` (capped at [`MAX_WAIT`]): until [`caught_up`],
/// the copy kept search-active so its scheduled refresh runs. A refresh is
/// never forced: a wait that expires is ES's timeout (`Some(reason)`), even
/// when the operations are processed but not yet refreshed. Blocking (the
/// single-node fleet search's job); the cluster polls [`caught_up`]
/// without holding a thread.
pub fn wait_refreshed(svc: &IndexService, shard: &IndexShard, checkpoint: i64, timeout: Duration) -> Option<String> {
    let timeout = timeout.min(MAX_WAIT);
    let deadline = Instant::now() + timeout;
    loop {
        if caught_up(svc, shard, checkpoint) {
            return None;
        }
        if Instant::now() >= deadline {
            return Some(wait_timed_out(checkpoint, timeout));
        }
        shard.shard.note_search_wait();
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// ES's reason for a `wait_for_checkpoints` timeout (`SearchService`:
/// `search_timeout_exception`, 429), the timeout as `TimeValue` renders.
pub fn wait_timed_out(checkpoint: i64, timeout: Duration) -> String {
    format!("Wait for seq_no [{checkpoint}] refreshed timed out [{}]", time_value(timeout))
}

/// ES `TimeValue.toString()`: the largest unit the value reaches, one
/// decimal truncated (`Strings.format1Decimals`: 1999ms is `1.9s`), a
/// zero decimal dropped.
fn time_value(d: Duration) -> String {
    let n = d.as_nanos().min(u64::MAX as u128) as u64;
    const UNITS: [(u64, &str); 6] = [(86_400_000_000_000, "d"), (3_600_000_000_000, "h"), (60_000_000_000, "m"), (1_000_000_000, "s"), (1_000_000, "ms"), (1_000, "micros")];
    if n == 0 {
        return "0s".into();
    }
    for (div, u) in UNITS {
        if n >= div {
            let (whole, tenth) = (n / div, (n % div) * 10 / div);
            return if tenth == 0 { format!("{whole}{u}") } else { format!("{whole}.{tenth}{u}") };
        }
    }
    format!("{n}nanos")
}

/// The `error` object ES renders for the shard failures of a
/// `wait_for_checkpoints` fleet search (`(shard, reason)`; `grouped`: one
/// entry per distinct reason) and its status.
pub fn shard_failure_error(index: &str, failures: &[(u32, String)], all_failed: bool) -> (u16, Value) {
    let mut grouped: Vec<&(u32, String)> = Vec::new();
    for f in failures {
        if !grouped.iter().any(|g| g.1 == f.1) {
            grouped.push(f);
        }
    }
    let iae = |r: &str| json!({"type": "illegal_argument_exception", "reason": r});
    let mut e = serde_json::Map::new();
    e.insert("root_cause".into(), Value::Array(grouped.iter().map(|(_, r)| iae(r)).collect()));
    e.insert("type".into(), json!("search_phase_execution_exception"));
    e.insert("reason".into(), json!(if all_failed { "all shards failed" } else { "Partial shards failure" }));
    e.insert("phase".into(), json!("query"));
    e.insert("grouped".into(), json!(true));
    e.insert("failed_shards".into(), Value::Array(grouped.iter().map(|(s, r)| json!({"shard": s, "index": index, "node": crate::node::NODE_ID, "reason": iae(r)})).collect()));
    if all_failed {
        if let Some((_, r)) = grouped.first() {
            let mut c = iae(r);
            c["caused_by"] = iae(r);
            e.insert("caused_by".into(), c);
        }
    }
    (400, Value::Object(e))
}

// ------------------------------------------------------------ secrets

pub const SECRETS_INDEX: &str = ".fleet-secrets-7";
pub const SECRETS_ALIAS: &str = ".fleet-secrets";

/// Shared standalone and cluster system-index definition.
pub fn secrets_index_definition() -> Value {
    json!({
        "settings": {"index": {"number_of_shards": 1, "auto_expand_replicas": "0-1", "hidden": true}},
        "mappings": {"dynamic": "false", "_meta": {"managed_index_mappings_version": 1, "version": crate::node::ES_VERSION},
                     "properties": {"value": {"type": "keyword", "index": false}}},
        "aliases": {SECRETS_ALIAS: {"is_hidden": true}},
    })
}

fn secrets_index(node: &Node, create: bool) -> Result<Option<Arc<IndexService>>> {
    if let Some(s) = node.get_index(SECRETS_INDEX) {
        return Ok(Some(s));
    }
    if !create {
        return Ok(None);
    }
    let body = secrets_index_definition();
    match node.create_index(SECRETS_INDEX, &body) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind == "resource_already_exists_exception" => Ok(node.get_index(SECRETS_INDEX)),
        Err(e) => Err(e),
    }
}

fn no_secret(id: &str) -> EsError {
    EsError::resource_not_found(format!("No secret with id [{id}]"))
}

/// `POST /_fleet/secret`: stores `value`, returns the new id.
pub fn post_secret(node: &Node, value: &str) -> Result<String> {
    let svc = secrets_index(node, true)?.ok_or_else(|| EsError::internal("cannot create fleet secrets index"))?;
    let id = crate::index::auto_id();
    let mut r = svc.write(vec![WriteReq::Index { id: Some(id.clone()), source: json!({"value": value}).to_string(), routing: None, cond: WriteCond::default() }], false);
    r.pop().ok_or_else(|| EsError::internal("no write result"))??;
    Ok(id)
}

/// `GET /_fleet/secret/{id}`.
pub fn get_secret(node: &Node, id: &str) -> Result<Value> {
    let svc = secrets_index(node, false)?.ok_or_else(|| no_secret(id))?;
    let g = svc.get(id)?.ok_or_else(|| no_secret(id))?;
    let src: Value = g.source.as_deref().and_then(|s| serde_json::from_str(s).ok()).unwrap_or(Value::Null);
    Ok(json!({"id": id, "value": src.get("value").cloned().unwrap_or(Value::Null)}))
}

/// `DELETE /_fleet/secret/{id}`.
pub fn delete_secret(node: &Node, id: &str) -> Result<()> {
    let svc = secrets_index(node, false)?.ok_or_else(|| no_secret(id))?;
    let mut r = svc.write(vec![WriteReq::Delete { id: id.to_string(), routing: None, cond: WriteCond::default() }], false);
    match r.pop().ok_or_else(|| EsError::internal("no write result"))? {
        Ok(w) if w.result == "deleted" => Ok(()),
        Ok(_) => Err(no_secret(id)),
        Err(e) if e.status == 404 => Err(no_secret(id)),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod time_value_tests {
    use super::*;

    #[test]
    fn time_values_truncate_like_es() {
        assert_eq!(time_value(Duration::from_millis(1999)), "1.9s");
        assert_eq!(time_value(Duration::from_secs(30)), "30s");
        assert_eq!(time_value(Duration::from_millis(500)), "500ms");
        assert_eq!(time_value(Duration::from_secs(90)), "1.5m");
        assert_eq!(time_value(Duration::from_secs(119)), "1.9m");
        assert_eq!(time_value(Duration::ZERO), "0s");
        assert_eq!(wait_timed_out(7, Duration::from_secs(30)), "Wait for seq_no [7] refreshed timed out [30s]");
    }
}
