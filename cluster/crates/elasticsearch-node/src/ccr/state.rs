//! CCR's cluster metadata (ES `AutoFollowMetadata`, the follower indices'
//! `ccr` custom metadata and the shard follow persistent tasks, in one
//! internal custom).

use crate::node::ClusterNode;
use elasticsearch_cluster::state::ClusterState;
use elasticsearch_engine::ccr::{AutoFollowPattern, FollowParams, FollowerMeta};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// The custom's key (internal: never in `GET _cluster/state`).
pub const CCR_CUSTOM: &str = "_ccr_state";

/// One follower index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Follower {
    /// The follower index's cluster uuid (empty while its restore creates it).
    #[serde(default)]
    pub uuid: String,
    pub meta: FollowerMeta,
    /// The user who followed (`StoredAuthn::encode`), sent with every
    /// leader request (ES keeps the user's headers in the shard follow task).
    #[serde(default)]
    pub identity: Option<String>,
    /// The resolved parameters of its shard follow tasks: present while
    /// following, absent while paused (or while it is being restored).
    #[serde(default)]
    pub params: Option<FollowParams>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CcrState {
    #[serde(default)]
    pub patterns: BTreeMap<String, AutoFollowPattern>,
    /// The identity of the user who put each pattern.
    #[serde(default)]
    pub pattern_identity: BTreeMap<String, Option<String>>,
    /// Pattern → leader index uuids already followed (or present when the
    /// pattern was put).
    #[serde(default)]
    pub followed: BTreeMap<String, Vec<String>>,
    /// By follower index name.
    #[serde(default)]
    pub followers: BTreeMap<String, Follower>,
}

impl CcrState {
    pub fn of(st: &ClusterState) -> CcrState {
        st.metadata.customs.get(CCR_CUSTOM).and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default()
    }

    fn is_empty(&self) -> bool {
        self.patterns.is_empty() && self.followed.is_empty() && self.followers.is_empty() && self.pattern_identity.is_empty()
    }

    /// The follower named `name` if it is that incarnation of the index.
    pub fn follower<'a>(&'a self, st: &ClusterState, name: &str) -> Option<&'a Follower> {
        let f = self.followers.get(name)?;
        let im = st.metadata.indices.get(name)?;
        (f.uuid.is_empty() || f.uuid == im.uuid).then_some(f)
    }

    /// Followers that are following, with their index's current uuid.
    pub fn following(&self, st: &ClusterState) -> Vec<(String, Follower)> {
        self.followers
            .iter()
            .filter(|(n, f)| f.params.is_some() && !f.uuid.is_empty() && st.metadata.indices.get(*n).is_some_and(|im| im.uuid == f.uuid))
            .map(|(n, f)| (n.clone(), f.clone()))
            .collect()
    }
}

/// Change the CCR state through the elected master (this node must be it).
/// `f` returns whether it changed anything.
pub async fn update<F>(node: &ClusterNode, f: F) -> Result<(), String>
where
    F: FnOnce(&mut CcrState) -> Result<bool, String> + Send + 'static,
{
    crate::snapshot_api::common::update_custom(node, CCR_CUSTOM, move |v: &mut Value| {
        let mut s: CcrState = serde_json::from_value(v.clone()).unwrap_or_default();
        if !f(&mut s)? {
            return Ok(false);
        }
        *v = if s.is_empty() { serde_json::json!({}) } else { serde_json::to_value(&s).map_err(|e| e.to_string())? };
        Ok(true)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_round_trips_and_empty_is_removed() {
        let mut s = CcrState::default();
        assert!(s.is_empty());
        s.followers.insert("f".into(), Follower {
            uuid: "u".into(),
            meta: FollowerMeta { follower_index: "f".into(), remote_cluster: "leader".into(), leader_index: "l".into(), leader_uuid: "lu".into(), leader_history_uuids: vec!["lu".into()], created_millis: 1 },
            identity: None,
            params: Some(FollowParams::default().resolve()),
        });
        let v = serde_json::to_value(&s).unwrap();
        let back: CcrState = serde_json::from_value(v).unwrap();
        assert_eq!(back.followers["f"].meta.leader_index, "l");
        assert!(back.followers["f"].params.as_ref().unwrap().max_read_request_operation_count == Some(5120));
    }
}
