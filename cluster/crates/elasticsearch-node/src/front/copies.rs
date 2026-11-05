//! Which shard copy serves a read, and what becomes of a shard no copy
//! could serve.
//!
//! * A data node answers a shard-level read only from a copy it holds
//!   readable ([`readable_copy`], ES `IndexShard.readAllowed`). A copy that
//!   is gone, being removed, reset for recovery or still recovering answers
//!   `shard_not_found_exception`, and the coordinator moves on.
//! * The coordinator orders each shard's copies by `preference`
//!   ([`Preference`], ES `OperationRouting.preferenceActiveShardIterator`)
//!   and tries them in turn on any failure. A shard that no copy answered is
//!   a shard failure ([`ShardFailure`]): partial results with
//!   `_shards.failures`, or ES's `search_phase_execution_exception` when
//!   every shard failed or partial results are not allowed
//!   (`AbstractSearchAsyncAction`).

use super::{RestReq, RestResp};
use crate::node::ClusterNode;
use crate::replication::recovery::RecoveryStage;
use elasticsearch_cluster::state::{ClusterState, ShardState};
use elasticsearch_engine::index::IndexShard;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

/// Internal request header of a shard-level read: the indices whose copy of
/// the request's shard (`SHARD_HEADER`, default 0) the receiving node must
/// hold readable. Stripped from client requests at the edge.
pub(super) const READ_COPIES_HEADER: &str = "x-rustic-internal-read-copies";

/// Name the copies a shard-level read reads (see [`READ_COPIES_HEADER`]).
pub(super) fn set_read_copies(headers: &mut Vec<(String, String)>, indices: &[String]) {
    headers.retain(|(k, _)| !k.eq_ignore_ascii_case(READ_COPIES_HEADER));
    headers.push((READ_COPIES_HEADER.into(), indices.join(",")));
}

pub(super) fn read_copies(headers: &[(String, String)]) -> Option<Vec<String>> {
    let (_, v) = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(READ_COPIES_HEADER))?;
    Some(v.split(',').filter(|s| !s.is_empty()).map(str::to_string).collect())
}

/// ES `ShardNotFoundException` for `[index][shard]` on this node.
pub(super) fn shard_not_found(node: &ClusterNode, index: &str, shard: u32) -> RestResp {
    let uuid = node.state().metadata.indices.get(index).map(|im| im.uuid.clone()).unwrap_or_else(|| "_na_".into());
    let e = elasticsearch_engine::error::EsError::shard_not_found(index, &uuid, shard);
    RestResp::json(e.status, &e.to_json())
}

/// The local copy of `[index][shard]` a read may use: the copy this node's
/// applied state assigns here as active (started, or relocating away), or
/// one assigned here that finished its recovery and waits for the master to
/// start it (ES `POST_RECOVERY` allows reads). A copy that is not in the
/// shard manager (removed), not open in its engine, closed, replaced, or
/// still recovering is not read.
pub(super) fn readable_copy(node: &ClusterNode, index: &str, shard: u32) -> Result<Arc<IndexShard>, RestResp> {
    let gone = || shard_not_found(node, index, shard);
    let Some(local) = node.shards.get(index, shard) else { return Err(gone()) };
    let (aid, svc, recovered) = {
        let g = local.lock();
        let recovered = g.recovery.as_ref().map_or(true, |slot| slot.lock().as_ref().is_some_and(|r| r.stage == RecoveryStage::Done));
        (g.allocation_id.clone(), g.svc.clone(), recovered)
    };
    let st = node.state();
    let me = node.local.id.as_str();
    let routing = st.routing_table.shard_table(index, shard)
        .and_then(|t| t.copies.iter().find(|c| c.node_id.as_deref() == Some(me) && c.allocation_id() == Some(aid.as_str())));
    let assigned = match routing {
        Some(c) if c.active() => true,
        Some(c) if c.state == ShardState::Initializing => recovered,
        _ => false,
    };
    if !assigned {
        return Err(gone());
    }
    // The engine's index service must be the copy's own (a wiped and
    // recreated shell is another copy), with this shard open on it.
    let current = node.engine_for(shard).and_then(|e| e.get_index(index)).is_some_and(|s| Arc::ptr_eq(&s, &svc));
    match svc.shard(shard).filter(|c| current && !c.is_closed()) {
        Some(copy) => Ok(copy),
        None => Err(gone()),
    }
}

/// The copies a shard-level read checked before it ran (see
/// [`READ_COPIES_HEADER`]); [`ReadGuard::held`] confirms afterwards that
/// none was replaced meanwhile (a reset copy reopened empty would otherwise
/// answer for the old one).
pub(super) struct ReadGuard {
    copies: Vec<(String, u32, Arc<IndexShard>)>,
}

impl ReadGuard {
    /// `None` for a request that names no copies. An ES|QL shard fetch
    /// names its shards in its fetch spec (any shard numbers); a read of a
    /// fold point in time it opened earlier reads that context, not a copy.
    pub(super) fn check(node: &ClusterNode, req: &RestReq) -> Result<Option<ReadGuard>, RestResp> {
        let wanted: Vec<(String, u32)> = match read_copies(&req.headers) {
            Some(indices) => {
                let shard = crate::shard_engines::shard_of(&req.headers).unwrap_or(0);
                indices.into_iter().map(|i| (i, shard)).collect()
            }
            None => match super::esql_dist::fetch_spec(req) {
                Some(spec) if spec.pit.as_ref().is_none_or(|p| p.phase == elasticsearch_engine::esql::remote::PitPhase::Open) => {
                    spec.shards.iter().map(|s| (s.index.clone(), s.shard)).collect()
                }
                _ => return Ok(None),
            },
        };
        let copies = wanted.into_iter().map(|(i, k)| readable_copy(node, &i, k).map(|c| (i, k, c))).collect::<Result<Vec<_>, _>>()?;
        Ok(Some(ReadGuard { copies }))
    }

    /// Another copy installed in its place, or its data wiped (a reset for
    /// re-recovery in place, an index shell recreated), fails the read. One
    /// that was detached meanwhile with its files intact (a relocation
    /// handoff, a removal) was read whole before it left (the engine fails
    /// a read that finds none), as ES completes a search on a shard that
    /// closes after it started.
    pub(super) fn held(&self, node: &ClusterNode) -> Result<(), RestResp> {
        for (index, shard, copy) in &self.copies {
            let svc = node.engine_for(*shard).and_then(|e| e.get_index(index));
            let replaced = match svc {
                None => true,
                Some(svc) => match svc.shard(*shard) {
                    Some(now) => !Arc::ptr_eq(&now, copy),
                    None => !copy.shard.path().exists(),
                },
            };
            if replaced {
                return Err(shard_not_found(node, index, *shard));
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------ preference

/// A parsed `preference` (ES `Preference`): an optional `_shards:` filter
/// and how each shard's copies are ordered.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Preference {
    pub shards: Option<BTreeSet<u32>>,
    pub kind: PreferenceKind,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum PreferenceKind {
    /// No preference: adaptive feedback ranks copies; ties rotate over requests.
    Rotate,
    /// `_local`: the local copy first.
    Local,
    /// `_only_local`: the local copy only.
    OnlyLocal,
    /// `_prefer_nodes:a,b`: copies on these nodes first.
    PreferNodes(String),
    /// `_only_nodes:spec`: copies on the matching nodes only.
    OnlyNodes(String),
    /// Any string not starting with `_`: the same copy for the same string,
    /// whichever node coordinates.
    Custom(String),
}

impl Preference {
    pub(super) fn parse(p: Option<&str>) -> Result<Preference, RestResp> {
        let Some(p) = p.filter(|p| !p.is_empty()) else { return Ok(Preference { shards: None, kind: PreferenceKind::Rotate }) };
        if !p.starts_with('_') {
            return Ok(Preference { shards: None, kind: PreferenceKind::Custom(p.to_string()) });
        }
        let (ty, value) = p.split_once(':').map_or((p, None), |(t, v)| (t, Some(v)));
        let bad = || {
            let e = elasticsearch_engine::error::EsError::illegal_argument(format!("no Preference for [{ty}]"));
            RestResp::json(e.status, &e.to_json())
        };
        let kind = match (ty, value) {
            ("_shards", Some(v)) => {
                let (list, rest) = v.split_once('|').map_or((v, None), |(l, r)| (l, Some(r)));
                let shards = list.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                let mut inner = Preference::parse(rest)?;
                inner.shards = Some(shards);
                return Ok(inner);
            }
            ("_local", None) => PreferenceKind::Local,
            ("_only_local", None) => PreferenceKind::OnlyLocal,
            ("_prefer_nodes", Some(v)) => PreferenceKind::PreferNodes(v.to_string()),
            ("_only_nodes", Some(v)) => PreferenceKind::OnlyNodes(v.to_string()),
            _ => return Err(bad()),
        };
        Ok(Preference { shards: None, kind })
    }

    /// Does this preference pin copies (anything but the default rotation)?
    pub(super) fn pins(&self) -> bool {
        self.kind != PreferenceKind::Rotate
    }
}

/// Java `String.hashCode` (over UTF-16 code units), as ES hashes ids.
fn java_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32))
}

/// The copy a custom preference string starts at among `n` copies (ES:
/// `Murmur3HashFunction.hash(preference)` mixed with the `ShardId` hash,
/// then a rotation of the shard's copy list by that seed).
fn custom_start(preference: &str, index: &str, uuid: &str, shard: u32, n: usize) -> usize {
    let index_hash = java_hash(index).wrapping_mul(31).wrapping_add(java_hash(uuid));
    let shard_hash = index_hash.wrapping_mul(31).wrapping_add(shard as i32);
    let seed = elasticsearch_cluster::routing::murmur3_hash(preference).wrapping_mul(31).wrapping_add(shard_hash);
    (seed as i64).rem_euclid(n as i64) as usize
}

/// The nodes to try for `[index][shard]`, in order, out of `holders` (the
/// nodes of its searchable copies). The order is the routing table's copy
/// order (primary first), the same on every node of one state, so a custom
/// preference picks the same copy from every coordinator. `rotation` spreads
/// requests without a preference; without one, adaptive replica selection
/// then ranks the copies, counting each shard's winner in `counts` (one
/// [`super::adaptive::Counts`] per search). Empty when the preference excludes every
/// copy (`_only_local` on a node without one); `_only_nodes` matching none is
/// ES's 400.
pub(super) fn order_copies(node: &ClusterNode, st: &ClusterState, pref: &Preference, index: &str, shard: u32, holders: &BTreeSet<String>, rotation: usize, counts: &mut super::adaptive::Counts) -> Result<Vec<String>, RestResp> {
    let mut copies: Vec<String> = Vec::new();
    for c in st.routing_table.shard_table(index, shard).into_iter().flat_map(|t| t.copies.iter()) {
        if let Some(n) = c.node_id.as_ref().filter(|n| holders.contains(*n) && !copies.contains(n)) {
            copies.push(n.clone());
        }
    }
    // Holders the table does not list (cannot happen for one state; kept so
    // no copy is ever dropped).
    let unlisted: Vec<String> = holders.iter().filter(|h| !copies.contains(h)).cloned().collect();
    copies.extend(unlisted);
    if copies.is_empty() {
        return Ok(copies);
    }
    let rotated = |start: usize| -> Vec<String> { (0..copies.len()).map(|i| copies[(start + i) % copies.len()].clone()).collect() };
    let first = |wanted: &BTreeSet<String>, rest: Vec<String>| -> Vec<String> {
        let (mut a, b): (Vec<String>, Vec<String>) = rest.into_iter().partition(|n| wanted.contains(n));
        a.extend(b);
        a
    };
    let me = node.local.id.clone();
    Ok(match &pref.kind {
        PreferenceKind::Rotate => {
            let mut ordered = rotated(rotation % copies.len());
            let enabled = st.metadata.setting("cluster.routing.use_adaptive_replica_selection")
                .or_else(|| node.config.settings.get("cluster.routing.use_adaptive_replica_selection"))
                .is_none_or(|value| value.as_bool().unwrap_or_else(|| value.as_str() != Some("false")));
            if enabled { node.adaptive_selection.order(st, &mut ordered, counts); }
            ordered
        },
        PreferenceKind::Local => first(&BTreeSet::from([me]), rotated(rotation % copies.len())),
        PreferenceKind::OnlyLocal => copies.iter().filter(|n| **n == me).cloned().collect(),
        PreferenceKind::PreferNodes(spec) => first(&crate::rest::nodes::resolve_nodes(st, &me, spec), rotated(rotation % copies.len())),
        PreferenceKind::OnlyNodes(spec) => {
            let nodes = crate::rest::nodes::resolve_nodes(st, &me, spec);
            let only: Vec<String> = rotated(rotation % copies.len()).into_iter().filter(|n| nodes.contains(n)).collect();
            // ES `onlyNodeSelectorActiveInitializingShardsIt`: the 400 only
            // when no active or initializing copy is on a matching node. One
            // that is not readable yet (initializing, or not yet started in
            // this node's view) leaves the shard without a copy: it fails
            // as unavailable, as ES's shard request to it would.
            let matching = st.routing_table.shard_table(index, shard).into_iter().flat_map(|t| t.copies.iter())
                .any(|c| (c.active() || c.state == ShardState::Initializing) && c.node_id.as_ref().is_some_and(|n| nodes.contains(n)));
            if only.is_empty() && matching {
                return Ok(only);
            }
            if only.is_empty() {
                let parts: Vec<&str> = spec.split(',').collect();
                let e = elasticsearch_engine::error::EsError::illegal_argument(format!(
                    "no data nodes with {} [{}] found for shard: [{index}][{shard}]", if parts.len() == 1 { "criteria" } else { "criterion" }, parts.join(",")));
                return Err(RestResp::json(e.status, &e.to_json()));
            }
            only
        }
        PreferenceKind::Custom(p) => {
            let uuid = st.metadata.indices.get(index).map(|im| im.uuid.as_str()).unwrap_or("_na_");
            rotated(custom_start(p, index, uuid, shard, copies.len()))
        }
    })
}

/// Every `"{index}#{shard}"` key's copies in the order they are tried
/// ([`order_copies`]). Keys the preference leaves no copy for are
/// `unavailable` (ES `failOnUnavailable`). One plan serves every phase of a
/// request (DFS, kNN, expand), so they start at the same copies; `pins`
/// records the copy each key's first answer came from, and later phases go
/// back to it.
pub(super) struct Plan {
    pub copies: BTreeMap<String, Vec<String>>,
    pub unavailable: Vec<(String, u32)>,
    pub pinned: bool,
    pins: parking_lot::Mutex<BTreeMap<String, String>>,
}

/// Split `"{index}#{shard}"`.
pub(super) fn split_key(key: &str) -> Option<(String, u32)> {
    key.rsplit_once('#').and_then(|(i, k)| Some((i.to_string(), k.parse().ok()?)))
}

impl Plan {
    /// `holders`: the searchable copies' nodes of each key.
    pub(super) fn new(node: &ClusterNode, req: &RestReq, holders: &BTreeMap<String, BTreeSet<String>>) -> Result<Plan, RestResp> {
        static NEXT_SEARCH_COPY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let pref = Preference::parse(req.params().get("preference").map(String::as_str))?;
        let rotation = NEXT_SEARCH_COPY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let st = node.state();
        let mut copies = BTreeMap::new();
        let mut unavailable = Vec::new();
        let mut counts = node.adaptive_selection.counts();
        // Adjacent shards start at different copies (`search_cover`).
        for (ordinal, (key, h)) in holders.iter().enumerate() {
            let Some((index, shard)) = split_key(key) else { continue };
            let order = order_copies(node, &st, &pref, &index, shard, h, rotation.wrapping_add(ordinal), &mut counts)?;
            if order.is_empty() {
                unavailable.push((index, shard));
            } else {
                copies.insert(key.clone(), order);
            }
        }
        Ok(Plan { copies, unavailable, pinned: pref.pins(), pins: Default::default() })
    }

    /// The nodes to try for `key`: its pinned copy alone once a phase got an
    /// answer from it, else its ordered copies.
    pub(super) fn candidates(&self, key: &str) -> Vec<String> {
        if let Some(n) = self.pins.lock().get(key) {
            return vec![n.clone()];
        }
        self.copies.get(key).cloned().unwrap_or_default()
    }

    pub(super) fn pin(&self, key: &str, node: &str) {
        self.pins.lock().entry(key.to_string()).or_insert_with(|| node.to_string());
    }

    /// The pins so far (see [`Plan::restore_pins`]).
    pub(super) fn pins(&self) -> BTreeMap<String, String> {
        self.pins.lock().clone()
    }

    /// Back to the pins an attempt started from: a phase that gave up (a
    /// query_then_fetch falling back to one round) must not leave its
    /// copies pinned for the attempt that replaces it, which then gets every
    /// copy to fall back on again. Pins of earlier phases stay.
    pub(super) fn restore_pins(&self, pins: BTreeMap<String, String>) {
        *self.pins.lock() = pins;
    }
}

// --------------------------------------------------------- shard failures

/// One shard of one index that no copy answered (ES `ShardSearchFailure`).
#[derive(Clone, Debug)]
pub(super) struct ShardFailure {
    pub index: String,
    pub shard: u32,
    pub node: Option<String>,
    pub status: u16,
    pub reason: Value,
    /// The engine's own one-shard `search_phase_execution_exception` the
    /// failure came in (its rendering of the cause is kept).
    pub engine_error: Option<Value>,
}

/// Error types that say the copy was not there to read (ES
/// `TransportActions.isShardNotAvailableException`); a later attempt's more
/// meaningful failure replaces them, never the other way round.
fn not_available(ty: &str) -> bool {
    matches!(ty, "shard_not_found_exception" | "index_not_found_exception" | "illegal_index_shard_state_exception" | "no_shard_available_action_exception" | "unavailable_shards_exception" | "already_closed_exception" | "index_shard_closed_exception")
}

fn error_type(r: &RestResp) -> String {
    r.body_json().and_then(|v| v["error"]["type"].as_str().map(str::to_string)).unwrap_or_default()
}

/// Whether a later copy's failed answer `next` replaces `prev` as the
/// shard's failure (see [`ShardFailure::replaced_by`]).
pub(super) fn replaces(prev: &RestResp, next: &RestResp) -> bool {
    !not_available(&error_type(next)) || not_available(&error_type(prev))
}

/// Transport-level failures: the node, not the copy, failed.
pub(super) fn node_failure(r: &RestResp) -> bool {
    let ty = r.body_json().and_then(|v| v["error"]["type"].as_str().map(str::to_string)).unwrap_or_default();
    matches!(ty.as_str(), "connect_transport_exception" | "node_not_connected_exception" | "node_disconnected_exception" | "transport_exception" | "receive_timeout_transport_exception" | "node_closed_exception")
}

/// A copy's answer, even when it is not a 2xx: a response that carries no
/// `error` (a document that is not there, a `HEAD`) is what every copy
/// would say.
pub(super) fn answered(r: &RestResp) -> bool {
    if r.is_success() {
        return true;
    }
    if r.body.is_empty() {
        return r.status < 500 && r.status != 429;
    }
    r.body_json().is_some_and(|v| v.get("error").is_none_or(Value::is_null))
}

/// A failed shard-level answer that no other copy can change: a request
/// error (parsing, validation, closed index) that ES raises once, before any
/// shard runs, and that the engine raises the same on every copy. Shard
/// failures (`search_phase_execution_exception`, script and query errors
/// included), rejections (429), timeouts, server errors and missing copies
/// are retried on the next copy.
pub(super) fn is_request_error(r: &RestResp) -> bool {
    if r.is_success() { return false; }
    let body = r.body_json();
    let ty = body.as_ref().and_then(|v| v["error"]["type"].as_str()).unwrap_or_default();
    // Inference endpoint lookup and compound-retriever failures happen
    // during request rewrite/ranking, not on an individual shard copy.
    // Retrying and wrapping these loses both their status and suppressed
    // rank-feature causes (a reranker may return a 500 status_exception).
    if ty == "resource_not_found_exception" || ty == "status_exception" {
        return true;
    }
    if !(400..500).contains(&r.status) || matches!(r.status, 404 | 408 | 429) {
        return false;
    }
    !(ty == "search_phase_execution_exception" || not_available(ty) || ty == "circuit_breaking_exception" || ty == "es_rejected_execution_exception")
}

impl ShardFailure {
    /// The failures a failed shard-level answer stands for, one per index
    /// the request covered (the engine wraps a shard's error in a one-shard
    /// `search_phase_execution_exception`; its cause is the failure).
    pub(super) fn from_response(r: &RestResp, indices: &[String], shard: u32, node: &str) -> Vec<ShardFailure> {
        let body = r.body_json().unwrap_or(Value::Null);
        let error = match &body["error"] {
            Value::Object(_) => body["error"].clone(),
            Value::String(s) => json!({"type": "exception", "reason": s}),
            _ => json!({"type": "exception", "reason": String::from_utf8_lossy(&r.body)}),
        };
        let wrapped = error["type"] == "search_phase_execution_exception";
        let failed = error["failed_shards"].as_array().cloned().unwrap_or_default();
        indices.iter().map(|index| {
            let reason = match failed.iter().find(|f| f["index"].as_str() == Some(index)).or_else(|| failed.first()) {
                Some(f) if wrapped && f["reason"].is_object() => f["reason"].clone(),
                _ => strip_root_cause(&error),
            };
            // ES AbstractSearchAsyncAction.onShardFailure wraps a copy that
            // disappeared after routing exactly like a shard with no candidate.
            // Never leak its internal 404 as the public search phase status.
            let unavailable = reason["type"].as_str().is_some_and(not_available);
            let reason = if unavailable {
                json!({"type": "no_shard_available_action_exception", "reason": reason.get("reason").cloned().unwrap_or(Value::Null)})
            } else { reason };
            ShardFailure { index: index.clone(), shard, node: (!node.is_empty()).then(|| node.to_string()),
                status: if unavailable { 503 } else { r.status }, reason,
                engine_error: (wrapped && !unavailable).then(|| error.clone()) }
        }).collect()
    }

    /// A shard with no copy to try (ES `failOnUnavailable`).
    pub(super) fn unavailable(index: &str, shard: u32) -> ShardFailure {
        ShardFailure { index: index.into(), shard, node: None, status: 503, reason: json!({"type": "no_shard_available_action_exception", "reason": null}), engine_error: None }
    }

    fn ty(&self) -> &str {
        self.reason["type"].as_str().unwrap_or_default()
    }

    /// Whether `next` (a later copy's failure) replaces this one.
    pub(super) fn replaced_by(&self, next: &ShardFailure) -> bool {
        !not_available(next.ty()) || not_available(self.ty())
    }

    pub(super) fn to_json(&self) -> Value {
        let index = if self.index.is_empty() { Value::Null } else { json!(self.index) };
        json!({"shard": self.shard, "index": index, "node": self.node, "reason": self.reason})
    }
}

fn strip_root_cause(error: &Value) -> Value {
    let mut e = error.clone();
    if let Some(o) = e.as_object_mut() {
        o.remove("root_cause");
        o.remove("stack_trace");
    }
    e
}

/// Failures in ES's shard order (`ShardId.compareTo`: shard number, then
/// index name; the search's shard iterators are sorted so, and its shard
/// failures listed by them). Stable: one shard's failures keep their order.
fn in_shard_order(failures: &[ShardFailure]) -> Vec<&ShardFailure> {
    let mut sorted: Vec<&ShardFailure> = failures.iter().collect();
    sorted.sort_by(|a, b| (a.shard, a.index.as_str()).cmp(&(b.shard, b.index.as_str())));
    sorted
}

/// ES `ExceptionsHelper.groupBy`: one failure per index and cause (type and
/// reason), the first of each, in shard order.
fn grouped(failures: &[ShardFailure]) -> Vec<&ShardFailure> {
    let mut seen = BTreeSet::new();
    in_shard_order(failures).into_iter().filter(|f| seen.insert((f.index.clone(), f.ty().to_string(), f.reason["reason"].to_string()))).collect()
}

/// Error types of plain Java (not Elasticsearch) exceptions, which ES
/// wraps anew when it looks for root causes.
fn java_exception(ty: &str) -> bool {
    matches!(ty, "illegal_argument_exception" | "illegal_state_exception" | "number_format_exception" | "null_pointer_exception" | "arithmetic_exception"
        | "unsupported_operation_exception" | "class_cast_exception" | "array_index_out_of_bounds_exception" | "index_out_of_bounds_exception"
        | "es_rejected_execution_exception" | "runtime_exception" | "io_exception" | "date_time_parse_exception" | "timeout_exception" | "too_complex_to_determinize_exception")
}

/// ES `guessRootCauses` of a failure's cause: the deepest Elasticsearch
/// exception of its `caused_by` chain (a plain Java cause stands for
/// itself), and whether that is the cause itself (the same object, which
/// `deduplicateCause` then leaves out as the phase failure's cause). It
/// renders as a root cause does: its own fields, no `caused_by`.
fn root_of(reason: &Value) -> (Value, bool) {
    let mut e = reason;
    let mut depth = 0;
    while e["caused_by"].is_object() && !java_exception(e["caused_by"]["type"].as_str().unwrap_or_default()) && !java_exception(e["type"].as_str().unwrap_or_default()) {
        e = &e["caused_by"];
        depth += 1;
    }
    let mut r = strip_root_cause(e);
    if let Some(o) = r.as_object_mut() {
        o.remove("caused_by");
    }
    let same = depth == 0 && !java_exception(reason["type"].as_str().unwrap_or_default());
    (r, same)
}

/// Add shard failures to a merged search / count response: `_shards.total`
/// and `failed` count them, `failures` lists them grouped (with the parts'
/// own failures).
pub(super) fn add_failures(v: &mut Value, failures: &[ShardFailure]) {
    if failures.is_empty() {
        return;
    }
    let Some(s) = v.get_mut("_shards").and_then(Value::as_object_mut) else { return };
    let n = failures.len() as u64;
    s.insert("total".into(), json!(s.get("total").and_then(Value::as_u64).unwrap_or(0) + n));
    s.insert("failed".into(), json!(s.get("failed").and_then(Value::as_u64).unwrap_or(0) + n));
    let mut list: Vec<Value> = s.remove("failures").and_then(|f| f.as_array().cloned()).unwrap_or_default();
    list.extend(grouped(failures).into_iter().map(ShardFailure::to_json));
    s.insert("failures".into(), Value::Array(list));
}

/// Add the failures of shards a merged response already counts successful
/// (a later phase, query_then_fetch's fetch, failed on them): they move from
/// `successful` to `failed`, and `failures` lists them grouped.
pub(super) fn add_lost_failures(v: &mut Value, lost: &[ShardFailure]) {
    if lost.is_empty() {
        return;
    }
    let Some(s) = v.get_mut("_shards").and_then(Value::as_object_mut) else { return };
    let n = lost.len() as u64;
    s.insert("successful".into(), json!(s.get("successful").and_then(Value::as_u64).unwrap_or(0).saturating_sub(n)));
    s.insert("failed".into(), json!(s.get("failed").and_then(Value::as_u64).unwrap_or(0) + n));
    let mut list: Vec<Value> = s.remove("failures").and_then(|f| f.as_array().cloned()).unwrap_or_default();
    list.extend(grouped(lost).into_iter().map(ShardFailure::to_json));
    s.insert("failures".into(), Value::Array(list));
}

/// A context page's `_shards` (scroll, PIT): `base` counts the shards the
/// context still reads; `uncounted` failed before `base` was taken (they
/// add to `total`), `lost` are shards of `base` whose context failed since
/// (no longer successful). Both are reported failed.
pub(super) fn context_shards(base: &Value, uncounted: &[ShardFailure], lost: &[ShardFailure]) -> Value {
    let mut s = base.clone();
    if uncounted.is_empty() && lost.is_empty() {
        return s;
    }
    let (u, l) = (uncounted.len() as u64, lost.len() as u64);
    s["total"] = json!(base["total"].as_u64().unwrap_or(0) + u);
    s["successful"] = json!(base["successful"].as_u64().unwrap_or(0).saturating_sub(l));
    s["failed"] = json!(base["failed"].as_u64().unwrap_or(0) + u + l);
    let all: Vec<ShardFailure> = uncounted.iter().chain(lost).cloned().collect();
    if let Some(o) = s.as_object_mut() {
        o.remove("failures");
        o.insert("failures".into(), Value::Array(grouped(&all).into_iter().map(ShardFailure::to_json).collect()));
    }
    s
}

/// The copy's node left while a request to it was in flight: ES's
/// `NodeDisconnectedException`, a `ConnectTransportException`, so 502 like
/// a failed connect (the transport's own table,
/// `TransportError::to_remote_error`). It was answered 500, which made a
/// search with `allow_partial_search_results=false` fail with 500 instead
/// of the retryable 502 a connect failure gets (c32i fault gate).
pub(super) fn node_disconnected(node_id: &str) -> RestResp {
    RestResp::error(502, "node_disconnected_exception", &format!("[{node_id}] disconnected"))
}

/// ES `SearchPhaseExecutionException` over `failures`: `reason` is
/// `all shards failed` or `Partial shards failure`. Its status is the first
/// failure's, unless a later one is a server error (`status()`); its cause
/// is the first failure's root cause unless that is the failure itself
/// (`deduplicateCause`).
pub(super) fn phase_failure(phase: &str, reason: &str, failures: &[ShardFailure]) -> RestResp {
    let groups = grouped(failures);
    // ES `SearchPhaseExecutionException.status`: over every shard failure
    // (not the grouped ones) in shard order, the first 502-504 one's at
    // once (an unavailable shard makes the search retryable), else the last
    // 5xx one's, else the first one's.
    let ordered = in_shard_order(failures);
    let mut status = ordered.first().map_or(503, |f| f.status);
    for f in &ordered {
        if (502..=504).contains(&f.status) {
            status = f.status;
            break;
        }
        if f.status >= 500 {
            status = f.status;
        }
    }
    let root_causes: Vec<Value> = groups.iter().flat_map(|f| match f.engine_error.as_ref().and_then(|e| e["root_cause"].as_array()) {
        Some(causes) => causes.clone(),
        None => vec![root_of(&f.reason).0],
    }).collect();
    let failed_shards: Vec<Value> = groups.iter().map(|f| f.to_json()).collect();
    // A failure the engine rendered keeps that rendering (its cause), with
    // the cluster's shards.
    if let Some(mut error) = groups.first().and_then(|f| f.engine_error.clone()) {
        error["root_cause"] = json!(root_causes);
        error["reason"] = json!(reason);
        error["phase"] = json!(phase);
        error["grouped"] = json!(true);
        error["failed_shards"] = json!(failed_shards);
        return RestResp::json(status, &json!({"error": error, "status": status}));
    }
    let mut error = json!({
        "root_cause": root_causes,
        "type": "search_phase_execution_exception",
        "reason": reason,
        "phase": phase,
        "grouped": true,
        "failed_shards": failed_shards,
    });
    if let Some((cause, false)) = groups.first().map(|f| root_of(&f.reason)) {
        error["caused_by"] = cause;
    }
    RestResp::json(status, &json!({"error": error, "status": status}))
}

/// `allow_partial_search_results` (default true).
pub(super) fn partial_allowed(params: &std::collections::HashMap<String, String>) -> bool {
    params.get("allow_partial_search_results").map(String::as_str) != Some("false")
}

/// `allow_partial_search_results`, else the cluster's
/// `search.default_allow_partial_results` (default true), as ES resolves it
/// for every search, scroll and point in time.
pub(super) fn allow_partial(node: &ClusterNode, params: &std::collections::HashMap<String, String>) -> bool {
    if params.contains_key("allow_partial_search_results") {
        return partial_allowed(params);
    }
    let key = "search.default_allow_partial_results";
    let cs = node.engine.cluster_settings.read();
    cs.1.get(key).or(cs.0.get(key)).map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).as_deref() != Some("false")
}

/// How long a shard-level read waits for one copy before it moves on to
/// the next.
///
/// No bound of its own: a search `timeout` is soft (the copy answers with
/// what it has and `timed_out: true`) and a scroll keep-alive is how long a
/// context lives, not how long a page may take; neither may turn into a
/// hard deadline that drops the answer and runs the shard again elsewhere.
/// A hung node is given up on when fault detection removes it
/// ([`node_left`]); the transport's forwarding deadline is the last resort.
pub(super) fn copy_bound(_params: &std::collections::HashMap<String, String>, _body: &Value) -> Option<Duration> {
    None
}

/// Resolves once `id` is no longer a node of the applied cluster state.
pub(super) async fn node_left(node: &ClusterNode, id: &str) {
    let mut rx = node.subscribe();
    loop {
        if !rx.borrow_and_update().nodes.contains_key(id) {
            return;
        }
        if rx.changed().await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fail(index: &str, shard: u32, status: u16, ty: &str, reason: &str) -> ShardFailure {
        ShardFailure { index: index.into(), shard, node: Some("n1".into()), status, reason: json!({"type": ty, "reason": reason}), engine_error: None }
    }

    #[test]
    fn explicit_preferences_do_not_use_adaptive_default_ordering() {
        for value in ["_local", "_only_local", "_prefer_nodes:n1", "_only_nodes:n1", "stable-user-key"] {
            assert_ne!(Preference::parse(Some(value)).unwrap().kind, PreferenceKind::Rotate);
        }
        // A shard restriction alone still uses the default adaptive ordering.
        assert_eq!(Preference::parse(Some("_shards:0")).unwrap().kind, PreferenceKind::Rotate);
    }

    #[test]
    fn preference_values_parse_like_es() {
        let p = |s: &str| Preference::parse(Some(s));
        assert_eq!(p("_local").unwrap().kind, PreferenceKind::Local);
        assert_eq!(p("_only_local").unwrap().kind, PreferenceKind::OnlyLocal);
        assert_eq!(p("_prefer_nodes:a,b").unwrap().kind, PreferenceKind::PreferNodes("a,b".into()));
        assert_eq!(p("_only_nodes:x").unwrap().kind, PreferenceKind::OnlyNodes("x".into()));
        assert_eq!(p("sess1").unwrap().kind, PreferenceKind::Custom("sess1".into()));
        let s = p("_shards:0,2|_local").unwrap();
        assert_eq!((s.shards, s.kind), (Some(BTreeSet::from([0, 2])), PreferenceKind::Local));
        let s = p("_shards:1|sess").unwrap();
        assert_eq!((s.shards, s.kind), (Some(BTreeSet::from([1])), PreferenceKind::Custom("sess".into())));
        assert_eq!(p("_shards:3").unwrap().kind, PreferenceKind::Rotate);
        let bad = p("_primary").unwrap_err();
        assert_eq!(bad.status, 400);
        assert_eq!(bad.body_json().unwrap()["error"]["reason"], "no Preference for [_primary]");
        assert!(p("_local:x").is_err());
        assert_eq!(Preference::parse(None).unwrap().kind, PreferenceKind::Rotate);
    }

    #[test]
    fn custom_preference_is_stable_and_spreads() {
        let a = custom_start("sess1", "i", "u", 0, 3);
        assert_eq!(a, custom_start("sess1", "i", "u", 0, 3));
        let starts: BTreeSet<usize> = (0..50).map(|i| custom_start(&format!("s{i}"), "i", "u", 0, 3)).collect();
        assert_eq!(starts.len(), 3, "different sessions land on every copy");
        assert!((0..20).all(|k| custom_start("p", "i", "u", k, 2) < 2));
        assert_eq!(java_hash("hello"), 99162322);
    }

    #[test]
    fn a_meaningful_failure_is_not_replaced_by_a_missing_copy() {
        let script = fail("i", 0, 400, "script_exception", "boom");
        let gone = fail("i", 0, 404, "shard_not_found_exception", "no such shard");
        assert!(!script.replaced_by(&gone));
        assert!(gone.replaced_by(&script));
        assert!(gone.replaced_by(&gone));
    }

    #[test]
    fn partial_failures_are_counted_and_grouped() {
        let mut v = json!({"_shards": {"total": 2, "successful": 2, "skipped": 0, "failed": 0}, "hits": {"hits": []}});
        add_failures(&mut v, &[fail("i", 3, 429, "circuit_breaking_exception", "x"), fail("i", 1, 429, "circuit_breaking_exception", "x"), fail("j", 0, 500, "exception", "y")]);
        assert_eq!(v["_shards"]["total"], 5);
        assert_eq!(v["_shards"]["successful"], 2);
        assert_eq!(v["_shards"]["failed"], 3);
        let f = v["_shards"]["failures"].as_array().unwrap();
        assert_eq!(f.len(), 2, "one per index and cause: {f:?}");
        // Shard order (shard, then index): j#0 before i#1.
        assert_eq!(f[0]["index"], "j");
        assert_eq!((f[1]["index"].as_str(), f[1]["shard"].as_u64()), (Some("i"), Some(1)));
    }

    /// Combined review r2: ES orders shard failures by `ShardId` (shard
    /// number first, then index), so qa#1's 400 after qb#0's 429 answers
    /// 429, caused by the trip; a 5xx anywhere still wins.
    #[test]
    fn all_shards_failed_status_follows_shard_id_order() {
        let r = phase_failure("query", "all shards failed", &[fail("qa", 1, 400, "query_shard_exception", "bad"), fail("qb", 0, 429, "circuit_breaking_exception", "[parent] Data too large")]);
        assert_eq!(r.status, 429);
        let v = r.body_json().unwrap();
        assert_eq!(v["error"]["root_cause"][0]["type"], "circuit_breaking_exception");
        assert_eq!(v["error"]["failed_shards"][0]["index"], "qb");
        // Same shard number: by index name.
        let r = phase_failure("query", "all shards failed", &[fail("qb", 0, 429, "circuit_breaking_exception", "x"), fail("qa", 0, 400, "query_shard_exception", "bad")]);
        assert_eq!(r.status, 400);
        // Status over every failure, not one per group: a later 5xx of a
        // group already seen still counts.
        let r = phase_failure("query", "all shards failed", &[fail("i", 0, 400, "exception", "x"), fail("i", 1, 400, "exception", "x"), fail("i", 2, 503, "exception", "x")]);
        assert_eq!(r.status, 503);
    }

    #[test]
    fn failures_of_a_later_phase_leave_successful() {
        let mut v = json!({"_shards": {"total": 3, "successful": 3, "skipped": 0, "failed": 0}});
        add_lost_failures(&mut v, &[fail("i", 2, 404, "search_context_missing_exception", "No search context found for id [x]")]);
        assert_eq!((v["_shards"]["total"].as_u64(), v["_shards"]["successful"].as_u64(), v["_shards"]["failed"].as_u64()), (Some(3), Some(2), Some(1)));
        assert_eq!(v["_shards"]["failures"][0]["shard"], 2);
    }

    #[test]
    fn copy_reset_after_selection_is_publicly_unavailable_not_not_found() {
        for kind in ["shard_not_found_exception", "illegal_index_shard_state_exception", "already_closed_exception"] {
            let response = RestResp::error(404, kind, "copy reset during recovery");
            let failures = ShardFailure::from_response(&response, &["inplace".into()], 2, "n2");
            let result = phase_failure("query", "all shards failed", &failures);
            let body = result.body_json().unwrap();
            assert_eq!(result.status, 503);
            assert_eq!(body["status"], 503);
            assert_eq!(body["error"]["type"], "search_phase_execution_exception");
            assert_eq!(body["error"]["failed_shards"][0]["reason"]["type"], "no_shard_available_action_exception");
            assert_eq!(body["error"]["failed_shards"][0]["shard"], 2);
            assert_eq!(body["error"]["failed_shards"][0]["node"], "n2");
            assert_eq!(body["error"]["failed_shards"][0]["reason"]["reason"], "copy reset during recovery");
            let meaningful = fail("inplace", 2, 429, "circuit_breaking_exception", "decode denied");
            assert!(!meaningful.replaced_by(&failures[0]), "a disappearing copy must not hide an earlier real failure");
            assert!(failures[0].replaced_by(&meaningful));
        }
    }

    /// A node that leaves mid-request fails the shard with the status of
    /// a connect failure (502, as the transport maps both), and so does the
    /// search phase over it, partial or not.
    #[test]
    fn node_disconnected_shard_failure_is_502_like_connect_transport() {
        let addr: std::net::SocketAddr = "127.0.0.1:9300".parse().unwrap();
        let transport = |e: elasticsearch_transport::TransportError| e.to_remote_error().status;
        let connect = transport(elasticsearch_transport::TransportError::ConnectTransport { addr, reason: "refused".into() });
        assert_eq!(transport(elasticsearch_transport::TransportError::NodeDisconnected { addr, action: "a".into() }), connect);
        let left = node_disconnected("qswk42CLQaqwF9lNjUdGfQ");
        assert_eq!(left.status, connect);
        let v = left.body_json().unwrap();
        assert_eq!(v["error"]["type"], "node_disconnected_exception");
        assert_eq!(v["error"]["reason"], "[qswk42CLQaqwF9lNjUdGfQ] disconnected");
        let failures = ShardFailure::from_response(&left, &["jepsen-b".into()], 0, "qswk42CLQaqwF9lNjUdGfQ");
        assert_eq!(failures[0].status, connect);
        for reason in ["Partial shards failure", "all shards failed"] {
            let phase = phase_failure("query", reason, &failures);
            assert_eq!(phase.status, 502, "{reason}");
            assert_eq!(phase.body_json().unwrap()["error"]["root_cause"][0]["type"], "node_disconnected_exception");
        }
    }

    #[test]
    fn all_shards_failed_has_es_shape_and_status() {
        let breaker = phase_failure("query", "all shards failed", &[fail("i", 0, 429, "circuit_breaking_exception", "[parent] Data too large")]);
        assert_eq!(breaker.status, 429);
        let v = breaker.body_json().unwrap();
        assert_eq!(v["error"]["type"], "search_phase_execution_exception");
        assert_eq!(v["error"]["reason"], "all shards failed");
        assert_eq!(v["error"]["root_cause"][0]["type"], "circuit_breaking_exception");
        assert_eq!(v["error"]["failed_shards"][0]["shard"], 0);
        assert!(v["error"].get("caused_by").is_none(), "the cause is the shard failure itself");
        assert_eq!(v["status"], 429);
        // A later server error decides the status.
        let mixed = phase_failure("query", "all shards failed", &[fail("i", 0, 400, "script_exception", "a"), fail("i", 1, 500, "exception", "b")]);
        assert_eq!(mixed.status, 500);
        // A 502-504 shard decides at once, over a later 500.
        let unavailable = phase_failure("query", "all shards failed", &[fail("i", 0, 503, "no_shard_available_action_exception", "a"), fail("i", 1, 500, "exception", "b")]);
        assert_eq!(unavailable.status, 503);
        let last = phase_failure("query", "all shards failed", &[fail("i", 0, 500, "exception", "a"), fail("i", 1, 501, "exception", "b"), fail("i", 2, 400, "x", "c")]);
        assert_eq!(last.status, 501);
        // The deepest Elasticsearch exception is the root cause; one below
        // the failure's own cause is the phase failure's cause too.
        let mut wrapped = fail("i", 0, 400, "search_exception", "outer");
        wrapped.reason["caused_by"] = json!({"type": "query_shard_exception", "reason": "failed to create query", "caused_by": {"type": "number_format_exception", "reason": "For input string: \"x\""}});
        let v = phase_failure("query", "Partial shards failure", &[wrapped]).body_json().unwrap();
        assert_eq!(v["error"]["reason"], "Partial shards failure");
        assert_eq!(v["error"]["root_cause"][0]["type"], "query_shard_exception");
        assert_eq!(v["error"]["caused_by"]["type"], "query_shard_exception");
        // A plain Java cause is wrapped anew: reported as the cause.
        let v = phase_failure("query", "all shards failed", &[fail("i", 0, 400, "illegal_argument_exception", "bad")]).body_json().unwrap();
        assert_eq!(v["error"]["caused_by"]["type"], "illegal_argument_exception");
        assert_eq!(v["error"]["root_cause"][0]["type"], "illegal_argument_exception");
        // An Elasticsearch cause over a Java one is its own root cause.
        let mut script = fail("i", 0, 400, "script_exception", "runtime error");
        script.reason["caused_by"] = json!({"type": "illegal_argument_exception", "reason": "no field"});
        let v = phase_failure("query", "all shards failed", &[script]).body_json().unwrap();
        assert_eq!(v["error"]["root_cause"][0]["type"], "script_exception");
        assert!(v["error"].get("caused_by").is_none());
    }

    #[test]
    fn engine_failures_unwrap_to_their_shard_cause() {
        let engine = RestResp::json(400, &json!({"error": {"root_cause": [{"type": "script_exception", "reason": "runtime error"}], "type": "search_phase_execution_exception", "reason": "all shards failed",
            "phase": "query", "grouped": true, "failed_shards": [{"shard": 0, "index": "i", "node": "x", "reason": {"type": "script_exception", "reason": "runtime error"}}]}, "status": 400}));
        let f = ShardFailure::from_response(&engine, &["i".into()], 2, "n2");
        assert_eq!((f[0].shard, f[0].node.as_deref(), f[0].reason["type"].as_str()), (2, Some("n2"), Some("script_exception")));
        // Every shard failed: the engine's rendering, with the cluster's shard and node.
        let all = phase_failure("query", "all shards failed", &f).body_json().unwrap();
        assert_eq!(all["error"]["root_cause"][0]["type"], "script_exception");
        assert_eq!(all["error"]["failed_shards"], json!([{"shard": 2, "index": "i", "node": "n2", "reason": {"type": "script_exception", "reason": "runtime error"}}]));
        assert_eq!(all["status"], 400);
        let breaker = RestResp::error(429, "circuit_breaking_exception", "[parent] Data too large");
        let f = ShardFailure::from_response(&breaker, &["a".into(), "b".into()], 0, "n1");
        assert_eq!(f.len(), 2);
        assert_eq!(f[1].reason["type"], "circuit_breaking_exception");
        assert!(f[1].reason.get("root_cause").is_none());
    }

    #[test]
    fn a_missing_document_is_an_answer() {
        assert!(answered(&RestResp::json(404, &json!({"_index": "i", "_id": "1", "found": false}))));
        assert!(answered(&RestResp::new(404,vec![],vec![])));
        assert!(!answered(&RestResp::new(503,vec![],vec![])));
        assert!(!answered(&RestResp::error(404, "shard_not_found_exception", "no such shard")));
        assert!(!answered(&RestResp::error(429, "circuit_breaking_exception", "x")));
    }

    #[test]
    fn only_request_errors_skip_the_next_copy() {
        assert!(is_request_error(&RestResp::error(400, "parsing_exception", "unknown query [x]")));
        assert!(!is_request_error(&RestResp::error(400, "search_phase_execution_exception", "all shards failed")));
        assert!(!is_request_error(&RestResp::error(429, "circuit_breaking_exception", "x")));
        assert!(!is_request_error(&RestResp::error(404, "shard_not_found_exception", "no such shard")));
        assert!(!is_request_error(&RestResp::error(500, "exception", "x")));
        assert!(!is_request_error(&RestResp::error(503, "no_shard_available_action_exception", "x")));
        assert!(is_request_error(&RestResp::error(404, "resource_not_found_exception", "Inference endpoint not found [missing]")));
        assert!(is_request_error(&RestResp::error(400, "status_exception", "invalid inference task type")));
        assert!(is_request_error(&RestResp::json(500, &json!({"error": {"type": "status_exception", "suppressed": [{"type": "search_phase_execution_exception", "caused_by": {"type": "illegal_state_exception"}}]}, "status": 500}))));
    }
}
