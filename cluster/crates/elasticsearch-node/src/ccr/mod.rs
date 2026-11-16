//! Cross-cluster replication between rustic clusters (x-pack `ccr`).
//!
//! * **State** ([`state`]): one internal metadata custom ([`CCR_CUSTOM`],
//!   hidden from `GET _cluster/state`) holds the auto-follow patterns (with
//!   the identity of the user who put each), the leader index uuids each
//!   pattern followed, every follower index's CCR metadata (by name, with
//!   its cluster uuid, its leader and the stored identity of the user who
//!   followed, ES's shard follow task headers) and the follow parameters of
//!   the followers that are following (absent while paused). It changes
//!   only through the elected master and is persisted with the cluster
//!   state, so following resumes after any restart.
//! * **Leader side** ([`leader`]): transport actions every node serves to
//!   remote clusters (allow-listed on remote-cluster connections) and to
//!   its own cluster's nodes (the routed hop): leader index metadata,
//!   shard changes by seq_no (long poll on the leader primary), retention
//!   leases (added / renewed / removed on every active copy of a leader
//!   shard: rustic leases are per copy, so a promoted replica keeps them),
//!   and restore sessions over a leader primary's pinned commit. A remote
//!   caller is authorized as the user who followed (its role names
//!   resolved by the leader cluster's own roles, ES's certificate-based
//!   remote cluster security).
//! * **Follower creation** ([`restore`]): `PUT {follower}/_ccr/follow`
//!   creates the follower with the leader's settings and mappings through
//!   the snapshot restore flow, from ES's `_ccr_<alias>` repository and
//!   snapshot `_latest_`: each follower primary fetches the leader primary's
//!   files over the remote connection, verifies them and adopts them as a
//!   snapshot file restore does; replicas recover from it.
//! * **Shard follow tasks** ([`follow`]): every node runs the tasks of the
//!   follower primaries it holds, reading the leader shard's operations and
//!   writing them to the follower primary with the leader's seq_nos and
//!   versions (replicated to the follower's replicas as any primary write).
//! * **Auto-follow** ([`auto_follow`]): the elected master follows new
//!   leader indices matching active patterns.
//! * **REST** ([`rest`]): the `_ccr` APIs.

pub mod auto_follow;
pub mod follow;
pub mod leader;
pub mod restore;
pub mod rest;
pub mod state;

use crate::front::{RestReq, RestResp};
use crate::node::ClusterNode;
use futures::future::BoxFuture;
use std::sync::Arc;

pub use state::{CcrState, CCR_CUSTOM};

/// Register CCR's leader-side actions (served to remote clusters) and start
/// the follower side's shard follow tasks and auto-follow coordinator.
pub fn register(node: &Arc<ClusterNode>) {
    leader::register(node);
    follow::register(node);
    auto_follow::register(node);
}

/// `/_ccr/*` and `/{index}/_ccr/*`: `None` when `req` is not a CCR API.
pub fn handle<'a>(node: &'a Arc<ClusterNode>, req: &'a RestReq) -> BoxFuture<'a, Option<RestResp>> {
    Box::pin(rest::handle(node, req))
}
