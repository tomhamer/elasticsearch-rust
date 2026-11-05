//! Public CCR requests cannot use a same-named local index as a remote leader.
//! Remote transport is currently unsupported, so dispatch directly to the
//! engine's explicit rejection instead of waiting for local shard allocation.
use super::*;

pub(super) async fn await_leader_copies(_node: &Arc<ClusterNode>, _req: &RestReq) {}
