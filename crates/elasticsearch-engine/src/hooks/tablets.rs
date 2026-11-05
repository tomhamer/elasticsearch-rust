//! Tablet metadata (sharding phase 1; tablets go live in phase 2).
//!
//! A logical shard (ES-visible, `number_of_shards`) owns the routing
//! partitions `[shard * f, (shard + 1) * f)` of `0..routing_num_shards`
//! (`f` = routing factor), exactly the partitions ES's
//! `IndexRouting` maps to it. A **tablet** owns a contiguous sub-range of
//! one shard's partitions; tablets are the unit of placement, replication
//! and recovery. Hashing is `elasticsearch_routing` (murmur3 over UTF-16,
//! `floorMod`), so a document's partition, shard and tablet agree with ES.
//!
//! **Vectors:** HNSW graphs are never rebuilt by merges. Splitting a tablet
//! moves documents, so each child needs its own graph: [`TabletMap::split`]
//! marks both children `vector_rebuild_pending`; the executor serves the
//! parent's graph with a partition-range filter until the background vector
//! optimizer ([`rustic_core::Index::optimize_vectors`], the only place
//! graphs are built) has built each child's, then calls
//! [`TabletMap::mark_vectors_rebuilt`].
//!
//! Default single-shard behaviour is untouched: nothing in the engine reads
//! tablet maps yet.

use crate::error::{EsError, Result};
use elasticsearch_routing::{murmur3_hash, validate_routing_shards, IndexRouting};
use serde::{Deserialize, Serialize};

/// Routing partitions `[start, end)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TabletRange {
    pub start: u32,
    pub end: u32,
}

impl TabletRange {
    pub fn new(start: u32, end: u32) -> TabletRange {
        TabletRange { start, end }
    }

    pub fn len(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, partition: u32) -> bool {
        self.start <= partition && partition < self.end
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TabletState {
    /// Serving reads and writes.
    Active,
    /// Being split / merged / moved: still authoritative for its range.
    Transitioning,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tablet {
    pub id: u64,
    /// Logical shard number.
    pub shard: u32,
    pub range: TabletRange,
    pub state: TabletState,
    /// The tablet's HNSW graphs must be (re)built in the background before
    /// it serves k-NN without its parent's graph.
    #[serde(default)]
    pub vector_rebuild_pending: bool,
}

/// The tablets of one index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TabletMap {
    pub index_uuid: String,
    pub number_of_shards: u32,
    pub routing_num_shards: u32,
    /// `index.routing_partition_size` (1 = unpartitioned).
    pub partition_size: u32,
    /// Sorted by `range.start`; ranges tile `0..routing_num_shards`.
    pub tablets: Vec<Tablet>,
    pub next_id: u64,
    /// Bumped on every change (publish/compare in cluster state).
    pub version: u64,
}

impl TabletMap {
    /// One tablet per logical shard. `routing_num_shards == 0` uses ES's
    /// default (`calculate_num_routing_shards`).
    pub fn new(index_uuid: &str, number_of_shards: u32, routing_num_shards: u32, partition_size: u32) -> Result<TabletMap> {
        let n = number_of_shards.max(1);
        let rns = if routing_num_shards == 0 { elasticsearch_routing::calculate_num_routing_shards(n) } else { routing_num_shards };
        validate_routing_shards(n, rns).map_err(EsError::illegal_argument)?;
        let f = rns / n;
        let tablets = (0..n).map(|s| Tablet { id: s as u64, shard: s, range: TabletRange::new(s * f, (s + 1) * f), state: TabletState::Active, vector_rebuild_pending: false }).collect();
        Ok(TabletMap { index_uuid: index_uuid.to_string(), number_of_shards: n, routing_num_shards: rns, partition_size: partition_size.max(1), tablets, next_id: n as u64, version: 1 })
    }

    pub fn routing(&self) -> IndexRouting {
        IndexRouting::new(self.number_of_shards, self.routing_num_shards, self.partition_size)
    }

    /// The routing partition of a document (`IndexRouting` before the
    /// division by the routing factor).
    pub fn partition(&self, id: &str, routing: Option<&str>) -> Result<u32> {
        let r = self.routing();
        let hash = if r.is_partitioned() {
            let routing = routing.ok_or_else(|| EsError::illegal_argument(elasticsearch_routing::RoutingError::RoutingRequired.to_string()))?;
            murmur3_hash(routing).wrapping_add(murmur3_hash(id).rem_euclid(r.partition_size as i32))
        } else {
            murmur3_hash(routing.unwrap_or(id))
        };
        Ok(hash.rem_euclid(self.routing_num_shards as i32) as u32)
    }

    pub fn shard_of_partition(&self, partition: u32) -> u32 {
        partition / self.routing().routing_factor()
    }

    pub fn tablet_of_partition(&self, partition: u32) -> Option<&Tablet> {
        let i = self.tablets.partition_point(|t| t.range.end <= partition);
        self.tablets.get(i).filter(|t| t.range.contains(partition))
    }

    /// The tablet a document lives in.
    pub fn tablet_for(&self, id: &str, routing: Option<&str>) -> Result<&Tablet> {
        let p = self.partition(id, routing)?;
        self.tablet_of_partition(p).ok_or_else(|| EsError::internal(format!("no tablet owns partition {p}")))
    }

    pub fn tablets_of_shard(&self, shard: u32) -> Vec<&Tablet> {
        self.tablets.iter().filter(|t| t.shard == shard).collect()
    }

    pub fn get(&self, id: u64) -> Option<&Tablet> {
        self.tablets.iter().find(|t| t.id == id)
    }

    /// Split a tablet at partition `at` (default: the middle). Returns the
    /// two new tablet ids; both need a vector rebuild.
    pub fn split(&mut self, tablet: u64, at: Option<u32>) -> Result<(u64, u64)> {
        let i = self.tablets.iter().position(|t| t.id == tablet).ok_or_else(|| EsError::resource_not_found(format!("no tablet [{tablet}]")))?;
        let t = self.tablets[i].clone();
        if t.range.len() < 2 {
            return Err(EsError::illegal_argument(format!("tablet [{tablet}] owns a single routing partition and cannot be split")));
        }
        let at = at.unwrap_or(t.range.start + t.range.len() / 2);
        if at <= t.range.start || at >= t.range.end {
            return Err(EsError::illegal_argument(format!("split point [{at}] is outside tablet [{tablet}] range [{}, {})", t.range.start, t.range.end)));
        }
        let (a, b) = (self.next_id, self.next_id + 1);
        self.next_id += 2;
        let child = |id, range| Tablet { id, shard: t.shard, range, state: TabletState::Active, vector_rebuild_pending: true };
        self.tablets.splice(i..=i, [child(a, TabletRange::new(t.range.start, at)), child(b, TabletRange::new(at, t.range.end))]);
        self.version += 1;
        Ok((a, b))
    }

    /// Merge two adjacent tablets of the same shard. The merged tablet
    /// needs a vector rebuild (graphs are never merged).
    pub fn merge(&mut self, left: u64, right: u64) -> Result<u64> {
        let i = self.tablets.iter().position(|t| t.id == left).ok_or_else(|| EsError::resource_not_found(format!("no tablet [{left}]")))?;
        let ok = self.tablets.get(i + 1).is_some_and(|r| r.id == right && r.shard == self.tablets[i].shard && r.range.start == self.tablets[i].range.end);
        if !ok {
            return Err(EsError::illegal_argument(format!("tablets [{left}] and [{right}] are not adjacent tablets of one shard")));
        }
        let id = self.next_id;
        self.next_id += 1;
        let m = Tablet { id, shard: self.tablets[i].shard, range: TabletRange::new(self.tablets[i].range.start, self.tablets[i + 1].range.end), state: TabletState::Active, vector_rebuild_pending: true };
        self.tablets.splice(i..=i + 1, [m]);
        self.version += 1;
        Ok(id)
    }

    pub fn mark_vectors_rebuilt(&mut self, tablet: u64) -> Result<()> {
        let t = self.tablets.iter_mut().find(|t| t.id == tablet).ok_or_else(|| EsError::resource_not_found(format!("no tablet [{tablet}]")))?;
        t.vector_rebuild_pending = false;
        self.version += 1;
        Ok(())
    }

    /// Tablets whose graphs still have to be built.
    pub fn pending_vector_rebuilds(&self) -> Vec<u64> {
        self.tablets.iter().filter(|t| t.vector_rebuild_pending).map(|t| t.id).collect()
    }

    /// Ranges tile `0..routing_num_shards` in order, and each tablet lies in
    /// its shard's partitions.
    pub fn validate(&self) -> Result<()> {
        let f = self.routing().routing_factor();
        let mut next = 0;
        for t in &self.tablets {
            if t.range.start != next || t.range.is_empty() {
                return Err(EsError::illegal_argument(format!("tablet [{}] range [{}, {}) does not continue at {next}", t.id, t.range.start, t.range.end)));
            }
            if t.range.start / f != t.shard || (t.range.end - 1) / f != t.shard {
                return Err(EsError::illegal_argument(format!("tablet [{}] crosses shard [{}]'s partitions", t.id, t.shard)));
            }
            next = t.range.end;
        }
        if next != self.routing_num_shards {
            return Err(EsError::illegal_argument(format!("tablets cover [0, {next}) of [0, {})", self.routing_num_shards)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tablets_agree_with_es_routing() {
        for (n, ps) in [(1u32, 1u32), (5, 1), (3, 1), (6, 3)] {
            let m = TabletMap::new("u", n, 0, ps).unwrap();
            m.validate().unwrap();
            let r = m.routing();
            for i in 0..300 {
                let id = format!("doc-{i}");
                let routing = if ps > 1 { Some("user-7") } else if i % 3 == 0 { Some("rk") } else { None };
                let t = m.tablet_for(&id, routing).unwrap();
                assert_eq!(t.shard, r.shard_id(&id, routing).unwrap(), "n={n} id={id}");
            }
        }
        let p = TabletMap::new("u", 6, 6, 3).unwrap();
        assert!(p.tablet_for("x", None).is_err(), "partitioned index needs routing");
    }

    #[test]
    fn split_and_merge_keep_routing() {
        let mut m = TabletMap::new("u", 2, 1024, 1).unwrap();
        let before: Vec<u32> = (0..200).map(|i| m.tablet_for(&i.to_string(), None).unwrap().shard).collect();
        let (a, b) = m.split(0, None).unwrap();
        let (c, _) = m.split(b, Some(400)).unwrap();
        m.validate().unwrap();
        assert_eq!(m.tablets.len(), 4);
        assert_eq!(m.pending_vector_rebuilds().len(), 3);
        let after: Vec<u32> = (0..200).map(|i| m.tablet_for(&i.to_string(), None).unwrap().shard).collect();
        assert_eq!(before, after, "splits never move a document to another logical shard");
        assert!(m.merge(a, 1).is_err(), "different shards");
        let merged = m.merge(a, c).unwrap();
        m.validate().unwrap();
        assert!(m.get(merged).unwrap().vector_rebuild_pending);
        m.mark_vectors_rebuilt(merged).unwrap();
        assert!(!m.get(merged).unwrap().vector_rebuild_pending);
        assert!(TabletMap::new("u", 3, 640, 1).is_err());
        let mut one = TabletMap::new("u", 1024, 1024, 1).unwrap();
        assert!(one.split(0, None).is_err());
        assert!(one.validate().is_ok());
    }
}
