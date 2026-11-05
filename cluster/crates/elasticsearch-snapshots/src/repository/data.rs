//! `RepositoryData`: the content of the root `index-N` blob.

use crate::snapshot::SnapshotState;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Generation of an empty repository (no `index-N` yet).
pub const EMPTY_REPO_GEN: i64 = -1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotEntry {
    pub name: String,
    pub uuid: String,
    /// `SnapshotState.value()`.
    pub state: u8,
    /// index_id → index metadata identifier.
    #[serde(default)]
    pub index_metadata_lookup: BTreeMap<String, String>,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub start_time_millis: i64,
    #[serde(default)]
    pub end_time_millis: i64,
    #[serde(default)]
    pub slm_policy: String,
}

impl SnapshotEntry {
    pub fn snapshot_state(&self) -> SnapshotState {
        SnapshotState::from_value(self.state)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexEntry {
    pub id: String,
    /// uuids of the snapshots containing this index.
    pub snapshots: Vec<String>,
    /// Current shard-level generation per shard (`indices/{id}/{shard}/index-{gen}`).
    #[serde(default)]
    pub shard_generations: Vec<Option<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepositoryData {
    #[serde(skip)]
    pub generation: i64,
    #[serde(default)]
    pub snapshots: Vec<SnapshotEntry>,
    /// index name → entry.
    #[serde(default)]
    pub indices: BTreeMap<String, IndexEntry>,
    /// metadata identifier → blob uuid (`indices/{id}/meta-{blob}.dat`).
    #[serde(default)]
    pub index_metadata_identifiers: BTreeMap<String, String>,
    #[serde(default)]
    pub min_version: Option<String>,
    #[serde(default)]
    pub uuid: String,
    #[serde(default)]
    pub cluster_id: String,
}

impl Default for RepositoryData {
    fn default() -> Self {
        Self::empty()
    }
}

impl RepositoryData {
    pub fn empty() -> Self {
        Self {
            generation: EMPTY_REPO_GEN,
            snapshots: Vec::new(),
            indices: BTreeMap::new(),
            index_metadata_identifiers: BTreeMap::new(),
            min_version: Some("7.9.0".into()),
            uuid: "_na_".into(),
            cluster_id: "_na_".into(),
        }
    }

    pub fn snapshot_by_name(&self, name: &str) -> Option<&SnapshotEntry> {
        self.snapshots.iter().find(|s| s.name == name)
    }

    pub fn snapshot_by_uuid(&self, uuid: &str) -> Option<&SnapshotEntry> {
        self.snapshots.iter().find(|s| s.uuid == uuid)
    }

    pub fn snapshot_names(&self) -> Vec<String> {
        self.snapshots.iter().map(|s| s.name.clone()).collect()
    }

    /// Index names contained in a snapshot, sorted.
    pub fn indices_of(&self, uuid: &str) -> Vec<String> {
        self.indices.iter().filter(|(_, e)| e.snapshots.iter().any(|s| s == uuid)).map(|(n, _)| n.clone()).collect()
    }

    pub fn index_by_id(&self, id: &str) -> Option<(&String, &IndexEntry)> {
        self.indices.iter().find(|(_, e)| e.id == id)
    }

    /// All metadata identifiers still referenced by any snapshot.
    pub fn referenced_identifiers(&self) -> BTreeSet<String> {
        self.snapshots.iter().flat_map(|s| s.index_metadata_lookup.values().cloned()).collect()
    }

    /// Add a finished snapshot. `indices` = (name, id, shard gens, metadata identifier).
    pub fn with_snapshot(
        &self,
        entry: SnapshotEntry,
        indices: &[(String, String, Vec<Option<String>>)],
        new_identifiers: &BTreeMap<String, String>,
    ) -> Self {
        let mut d = self.clone();
        for (name, id, gens) in indices {
            let e = d.indices.entry(name.clone()).or_insert_with(|| IndexEntry {
                id: id.clone(),
                snapshots: Vec::new(),
                shard_generations: Vec::new(),
            });
            if !e.snapshots.contains(&entry.uuid) {
                e.snapshots.push(entry.uuid.clone());
            }
            e.shard_generations = gens.clone();
        }
        for (k, v) in new_identifiers {
            d.index_metadata_identifiers.insert(k.clone(), v.clone());
        }
        d.snapshots.push(entry);
        d
    }

    /// Remove snapshots; returns the new data and the ids of indices that are
    /// no longer referenced by any snapshot. Unreferenced metadata
    /// identifiers are dropped.
    pub fn without_snapshots(&self, uuids: &BTreeSet<String>) -> (Self, Vec<(String, IndexEntry)>) {
        let mut d = self.clone();
        d.snapshots.retain(|s| !uuids.contains(&s.uuid));
        let mut removed = Vec::new();
        let names: Vec<String> = d.indices.keys().cloned().collect();
        for name in names {
            let e = d.indices.get_mut(&name).unwrap();
            e.snapshots.retain(|s| !uuids.contains(s));
            if e.snapshots.is_empty() {
                removed.push((name.clone(), d.indices.remove(&name).unwrap()));
            }
        }
        let referenced = d.referenced_identifiers();
        d.index_metadata_identifiers.retain(|k, _| referenced.contains(k));
        (d, removed)
    }

    pub fn to_json_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("serializable")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_remove() {
        let d = RepositoryData::empty();
        let e = SnapshotEntry {
            name: "s1".into(),
            uuid: "u1".into(),
            state: 1,
            index_metadata_lookup: [("id1".to_string(), "ident1".to_string())].into(),
            version: crate::snapshot::ES_VERSION.into(),
            start_time_millis: 1,
            end_time_millis: 2,
            slm_policy: String::new(),
        };
        let d = d.with_snapshot(
            e,
            &[("idx".into(), "id1".into(), vec![Some("g1".into())])],
            &[("ident1".to_string(), "blob1".to_string())].into(),
        );
        assert_eq!(d.indices_of("u1"), vec!["idx"]);
        let json = String::from_utf8(d.to_json_bytes()).unwrap();
        assert!(json.contains("\"index_metadata_identifiers\":{\"ident1\":\"blob1\"}"));
        let (d2, removed) = d.without_snapshots(&["u1".to_string()].into());
        assert!(d2.snapshots.is_empty());
        assert_eq!(removed[0].0, "idx");
        assert!(d2.index_metadata_identifiers.is_empty());
    }
}
