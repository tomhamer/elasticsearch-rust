//! Binary-safe cluster state representation and publication diffs.
//!
//! The cluster state embeds `serde_json::Value` (mappings, settings,
//! templates, customs), which a non-self-describing format such as bincode
//! cannot decode (`Value` needs `deserialize_any`). Those parts are carried
//! as pre-serialized JSON bytes in binary formats and in their natural
//! form in human-readable ones (the persisted `coordination.json`, REST
//! output), so the JSON shape of the state is unchanged:
//!
//! * [`json_bytes`] is a `#[serde(with = ...)]` adapter for plain fields;
//! * [`Shared`] holds a value behind an `Arc` with its encoded bytes
//!   cached: cloning a state does not copy it, and an unchanged value is
//!   encoded once, however many publications and nodes it goes to.
//!
//! [`ClusterStateDiff`] is what a master publishes to a node that accepted
//! the previous state: only the indices, routing tables, templates and
//! customs that changed, plus the small always-sent parts. A node applies
//! it to its last accepted state only if that is exactly the diff's base
//! (version and state uuid); otherwise it asks for the full state.

use crate::state::{ClusterBlocks, ClusterState, CoordinationMetadata, DiscoveryNode, IndexMetadata, IndexRoutingTable};
use serde::de::{DeserializeOwned, Error as _};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

struct BytesVisitor;

impl<'de> serde::de::Visitor<'de> for BytesVisitor {
    type Value = Vec<u8>;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("JSON bytes")
    }
    fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Vec<u8>, E> {
        Ok(v.to_vec())
    }
    fn visit_byte_buf<E: serde::de::Error>(self, v: Vec<u8>) -> Result<Vec<u8>, E> {
        Ok(v)
    }
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<u8>, A::Error> {
        let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(1 << 20));
        while let Some(b) = seq.next_element::<u8>()? {
            out.push(b);
        }
        Ok(out)
    }
}

fn read_bytes<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
    d.deserialize_byte_buf(BytesVisitor)
}

/// `#[serde(with = "json_bytes")]`: the field as is in human-readable
/// formats, as JSON bytes in binary ones.
pub mod json_bytes {
    use super::*;

    pub fn serialize<T: Serialize, S: Serializer>(v: &T, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            v.serialize(s)
        } else {
            let bytes = serde_json::to_vec(v).map_err(serde::ser::Error::custom)?;
            s.serialize_bytes(&bytes)
        }
    }

    pub fn deserialize<'de, T: DeserializeOwned, D: Deserializer<'de>>(d: D) -> Result<T, D::Error> {
        if d.is_human_readable() {
            T::deserialize(d)
        } else {
            let bytes = read_bytes(d)?;
            serde_json::from_slice(&bytes).map_err(D::Error::custom)
        }
    }
}

struct SharedInner<T> {
    value: T,
    /// The JSON encoding, made at most once per value.
    bytes: OnceLock<Arc<[u8]>>,
}

impl<T: Clone> Clone for SharedInner<T> {
    fn clone(&self) -> Self {
        SharedInner { value: self.value.clone(), bytes: OnceLock::new() }
    }
}

/// An immutable-by-default shared value (see the module docs). Reads go
/// through `Deref`; a write (`DerefMut`) copies the value first if another
/// state still shares it, and drops the cached encoding.
pub struct Shared<T>(Arc<SharedInner<T>>);

impl<T> Shared<T> {
    pub fn new(value: T) -> Shared<T> {
        Shared(Arc::new(SharedInner { value, bytes: OnceLock::new() }))
    }

    /// Whether both hold the very same value (no comparison needed).
    pub fn ptr_eq(&self, other: &Shared<T>) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// The JSON encoding of the value, cached.
    pub fn json_bytes(&self) -> Result<Arc<[u8]>, serde_json::Error>
    where
        T: Serialize,
    {
        if let Some(b) = self.0.bytes.get() {
            return Ok(b.clone());
        }
        let b: Arc<[u8]> = serde_json::to_vec(&self.0.value)?.into();
        let _ = self.0.bytes.set(b.clone());
        Ok(b)
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        Shared(self.0.clone())
    }
}

impl<T> std::ops::Deref for Shared<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0.value
    }
}

impl<T: Clone> std::ops::DerefMut for Shared<T> {
    fn deref_mut(&mut self) -> &mut T {
        let inner = Arc::make_mut(&mut self.0);
        inner.bytes = OnceLock::new();
        &mut inner.value
    }
}

impl<T: PartialEq> PartialEq for Shared<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other) || self.0.value == other.0.value
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.value.fmt(f)
    }
}

impl<T: Default> Default for Shared<T> {
    fn default() -> Self {
        Shared::new(T::default())
    }
}

impl<T> From<T> for Shared<T> {
    fn from(value: T) -> Self {
        Shared::new(value)
    }
}

impl<T: Serialize> Serialize for Shared<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            self.0.value.serialize(s)
        } else {
            let bytes = self.json_bytes().map_err(serde::ser::Error::custom)?;
            s.serialize_bytes(&bytes)
        }
    }
}

impl<'de, T: DeserializeOwned> Deserialize<'de> for Shared<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            return T::deserialize(d).map(Shared::new);
        }
        let bytes = read_bytes(d)?;
        let value: T = serde_json::from_slice(&bytes).map_err(D::Error::custom)?;
        // Keep the received encoding: a node that forwards the value (or
        // the state) does not encode it again.
        let cell = OnceLock::new();
        let _ = cell.set(Arc::<[u8]>::from(bytes));
        Ok(Shared(Arc::new(SharedInner { value, bytes: cell })))
    }
}

// ---------------------------------------------------------------- diffs ----

fn map_diff<V: Clone + PartialEq>(base: &BTreeMap<String, V>, new: &BTreeMap<String, V>) -> BTreeMap<String, Option<V>> {
    let mut d = BTreeMap::new();
    for (k, v) in new {
        if base.get(k) != Some(v) {
            d.insert(k.clone(), Some(v.clone()));
        }
    }
    for k in base.keys() {
        if !new.contains_key(k) {
            d.insert(k.clone(), None);
        }
    }
    d
}

fn map_apply<V: Clone>(d: &BTreeMap<String, Option<V>>, m: &mut BTreeMap<String, V>) {
    for (k, v) in d {
        match v {
            Some(v) => {
                m.insert(k.clone(), v.clone());
            }
            None => {
                m.remove(k);
            }
        }
    }
}

/// The metadata part of a [`ClusterStateDiff`]: per-index, per-template and
/// per-custom upserts (`Some`) and removals (`None`); settings when changed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetadataDiff {
    pub cluster_uuid: String,
    pub cluster_uuid_committed: bool,
    pub version: u64,
    #[serde(with = "json_bytes")]
    pub persistent_settings: Option<Map<String, Value>>,
    #[serde(with = "json_bytes")]
    pub transient_settings: Option<Map<String, Value>>,
    pub indices: BTreeMap<String, Option<IndexMetadata>>,
    #[serde(with = "json_bytes")]
    pub index_templates: BTreeMap<String, Option<Value>>,
    #[serde(with = "json_bytes")]
    pub component_templates: BTreeMap<String, Option<Value>>,
    #[serde(with = "json_bytes")]
    pub legacy_templates: BTreeMap<String, Option<Value>>,
    #[serde(with = "json_bytes")]
    pub customs: BTreeMap<String, Option<Value>>,
}

/// A published state expressed against the previous one (ES's
/// `ClusterState.diff`). Nodes, coordination metadata and blocks are small
/// and sent whole (nodes and blocks only when changed).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterStateDiff {
    pub base_version: u64,
    pub base_state_uuid: String,
    pub cluster_name: String,
    pub version: u64,
    pub state_uuid: String,
    pub master_node_id: Option<String>,
    pub nodes: Option<BTreeMap<String, DiscoveryNode>>,
    pub coordination: CoordinationMetadata,
    pub metadata: MetadataDiff,
    pub routing_version: u64,
    pub routing_indices: BTreeMap<String, Option<IndexRoutingTable>>,
    pub blocks: Option<ClusterBlocks>,
}

impl ClusterStateDiff {
    /// The diff that turns `base` into `new`.
    pub fn between(base: &ClusterState, new: &ClusterState) -> ClusterStateDiff {
        let (bm, nm) = (&base.metadata, &new.metadata);
        ClusterStateDiff {
            base_version: base.version,
            base_state_uuid: base.state_uuid.clone(),
            cluster_name: new.cluster_name.clone(),
            version: new.version,
            state_uuid: new.state_uuid.clone(),
            master_node_id: new.master_node_id.clone(),
            nodes: (base.nodes != new.nodes).then(|| new.nodes.clone()),
            coordination: new.coordination.clone(),
            metadata: MetadataDiff {
                cluster_uuid: nm.cluster_uuid.clone(),
                cluster_uuid_committed: nm.cluster_uuid_committed,
                version: nm.version,
                persistent_settings: (bm.persistent_settings != nm.persistent_settings).then(|| nm.persistent_settings.clone()),
                transient_settings: (bm.transient_settings != nm.transient_settings).then(|| nm.transient_settings.clone()),
                indices: map_diff(&bm.indices, &nm.indices),
                index_templates: map_diff(&bm.index_templates, &nm.index_templates),
                component_templates: map_diff(&bm.component_templates, &nm.component_templates),
                legacy_templates: map_diff(&bm.legacy_templates, &nm.legacy_templates),
                customs: map_diff(&bm.customs, &nm.customs),
            },
            routing_version: new.routing_table.version,
            routing_indices: map_diff(&base.routing_table.indices, &new.routing_table.indices),
            blocks: (base.blocks != new.blocks).then(|| new.blocks.clone()),
        }
    }

    /// Whether `base` is the state this diff was computed against.
    pub fn applies_to(&self, base: &ClusterState) -> bool {
        base.version == self.base_version && base.state_uuid == self.base_state_uuid
    }

    /// Rebuild the full published state from its base.
    pub fn apply(&self, base: &ClusterState) -> Result<ClusterState, String> {
        if !self.applies_to(base) {
            return Err(format!(
                "cluster state diff is based on version [{}] uuid [{}], local state is version [{}] uuid [{}]",
                self.base_version, self.base_state_uuid, base.version, base.state_uuid
            ));
        }
        let mut s = base.clone();
        s.cluster_name = self.cluster_name.clone();
        s.version = self.version;
        s.state_uuid = self.state_uuid.clone();
        s.master_node_id = self.master_node_id.clone();
        if let Some(n) = &self.nodes {
            s.nodes = n.clone();
        }
        s.coordination = self.coordination.clone();
        let md = &self.metadata;
        let m = &mut s.metadata;
        m.cluster_uuid = md.cluster_uuid.clone();
        m.cluster_uuid_committed = md.cluster_uuid_committed;
        m.version = md.version;
        if let Some(p) = &md.persistent_settings {
            m.persistent_settings = p.clone();
        }
        if let Some(t) = &md.transient_settings {
            m.transient_settings = t.clone();
        }
        map_apply(&md.indices, &mut m.indices);
        map_apply(&md.index_templates, &mut m.index_templates);
        map_apply(&md.component_templates, &mut m.component_templates);
        map_apply(&md.legacy_templates, &mut m.legacy_templates);
        map_apply(&md.customs, &mut m.customs);
        s.routing_table.version = self.routing_version;
        map_apply(&self.routing_indices, &mut s.routing_table.indices);
        if let Some(b) = &self.blocks {
            s.blocks = b.clone();
        }
        Ok(s)
    }

    /// Indices whose metadata the diff carries (for logging / tests).
    pub fn changed_indices(&self) -> usize {
        self.metadata.indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{IndexShardRoutingTable, IndexState, RecoverySource, ShardRouting, UnassignedInfo, UnassignedReason};
    use serde_json::json;
    use std::collections::BTreeSet;

    fn index(name: &str, fields: usize) -> IndexMetadata {
        let mut props = Map::new();
        for i in 0..fields {
            props.insert(format!("f{i}"), json!({"type": "keyword"}));
        }
        let mut settings = Map::new();
        settings.insert("index.number_of_shards".into(), json!("1"));
        IndexMetadata {
            name: name.into(), uuid: format!("{name}-uuid"), version: 1, mapping_version: 1, settings_version: 1, aliases_version: 1,
            state: IndexState::Open, settings: settings.into(), mappings: Shared::new(json!({"properties": props})), aliases: BTreeMap::new(),
            number_of_shards: 1, number_of_replicas: 1, routing_num_shards: 1, primary_terms: vec![1],
            in_sync_allocation_ids: BTreeMap::from([(0, BTreeSet::from(["a".to_string()]))]), creation_date: 7, lifecycle: None,
        }
    }

    fn state() -> ClusterState {
        let mut s = ClusterState::empty("c");
        s.version = 10;
        for i in 0..20 {
            let name = format!("i{i}");
            s.metadata.indices.insert(name.clone(), index(&name, 50));
            let info = UnassignedInfo { reason: UnassignedReason::IndexCreated, at_millis: 1, failed_allocations: 0, delayed: false, details: None, last_allocation_status: "no_attempt".into(), last_node: None };
            let copy = ShardRouting::unassigned(&name, 0, true, info, Some(RecoverySource::ExistingStore { bootstrap_new_history_uuid: false }));
            s.routing_table.indices.insert(name.clone(), IndexRoutingTable { index_uuid: format!("{name}-uuid"), shards: vec![IndexShardRoutingTable { shard: 0, copies: vec![copy] }] });
        }
        s.metadata.customs.insert("ingest".into(), json!({"pipeline": []}));
        s.metadata.persistent_settings.insert("cluster.x".into(), json!("1"));
        s
    }

    #[test]
    fn full_state_round_trips_through_bincode_and_json() {
        let s = state();
        let bytes = bincode::serialize(&s).expect("bincode encodes the state");
        let back: ClusterState = bincode::deserialize(&bytes).expect("bincode decodes the state");
        assert_eq!(back, s);
        // The human-readable shape is unchanged (persisted state, REST).
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["metadata"]["indices"]["i0"]["mappings"]["properties"]["f0"]["type"], "keyword");
        assert_eq!(v["routing_table"]["indices"]["i0"]["shards"][0]["copies"][0]["recovery_source"]["type"], "EXISTING_STORE");
        let back: ClusterState = serde_json::from_value(v).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn cloned_states_share_mappings_until_written() {
        let s = state();
        let mut next = s.next();
        assert!(next.metadata.indices["i0"].mappings.ptr_eq(&s.metadata.indices["i0"].mappings));
        next.metadata.indices.get_mut("i0").unwrap().mappings["properties"]["new"] = json!({"type": "long"});
        assert!(!next.metadata.indices["i0"].mappings.ptr_eq(&s.metadata.indices["i0"].mappings));
        assert!(s.metadata.indices["i0"].mappings["properties"].get("new").is_none(), "copy on write");
    }

    #[test]
    fn diff_carries_only_changes_and_round_trips() {
        let base = state();
        let mut new = base.next();
        new.metadata.version += 1;
        new.metadata.indices.get_mut("i3").unwrap().mappings["properties"]["added"] = json!({"type": "long"});
        new.metadata.indices.get_mut("i3").unwrap().version += 1;
        new.metadata.indices.remove("i4");
        new.routing_table.indices.remove("i4");
        new.metadata.indices.insert("fresh".into(), index("fresh", 3));
        new.metadata.customs.remove("ingest");
        new.metadata.customs.insert("x".into(), json!({"y": 1}));
        new.routing_table.version += 1;
        let d = ClusterStateDiff::between(&base, &new);
        assert_eq!(d.changed_indices(), 3, "i3 changed, i4 removed, fresh added; 17 untouched indices are not sent");
        assert!(d.nodes.is_none() && d.blocks.is_none() && d.metadata.persistent_settings.is_none());
        let bytes = bincode::serialize(&d).unwrap();
        let full = bincode::serialize(&new).unwrap();
        assert!(bytes.len() * 4 < full.len(), "diff {} bytes vs full {} bytes", bytes.len(), full.len());
        let d: ClusterStateDiff = bincode::deserialize(&bytes).unwrap();
        let rebuilt = d.apply(&base).unwrap();
        assert_eq!(rebuilt, new);
        // Untouched indices keep the receiver's own (shared) mappings.
        assert!(rebuilt.metadata.indices["i0"].mappings.ptr_eq(&base.metadata.indices["i0"].mappings));
    }

    #[test]
    fn diff_refuses_a_different_base() {
        let base = state();
        let new = base.next();
        let d = ClusterStateDiff::between(&base, &new);
        let mut other = base.clone();
        other.state_uuid = "other".into();
        assert!(d.apply(&other).is_err());
        let mut older = base.clone();
        older.version -= 1;
        assert!(!d.applies_to(&older));
    }
}
