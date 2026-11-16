//! Index tombstones (ES `IndexGraveyard`): the metadata custom
//! `index-graveyard` records the last `cluster.indices.tombstones.size`
//! (500) deleted indices, so a node that was away when an index was deleted
//! deletes its copy when it rejoins, while a local index the metadata does
//! not know and no tombstone names is kept as a dangling index.
//!
//! Wire/XContent form (as ES renders it):
//! `{"tombstones": [{"index": {"index_name": "i", "index_uuid": "u"}, "delete_date_in_millis": 1}]}`.

use crate::state::Metadata;
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// The metadata custom holding the graveyard.
pub const CUSTOM: &str = "index-graveyard";
/// `cluster.indices.tombstones.size` default.
pub const DEFAULT_SIZE: usize = 500;

/// One deleted index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tombstone {
    pub index_name: String,
    pub index_uuid: String,
    pub delete_date_in_millis: i64,
}

impl Tombstone {
    fn to_json(&self) -> Value {
        json!({"index": {"index_name": self.index_name, "index_uuid": self.index_uuid}, "delete_date_in_millis": self.delete_date_in_millis})
    }

    fn from_json(v: &Value) -> Option<Tombstone> {
        Some(Tombstone {
            index_name: v["index"]["index_name"].as_str()?.to_string(),
            index_uuid: v["index"]["index_uuid"].as_str()?.to_string(),
            delete_date_in_millis: v["delete_date_in_millis"].as_i64().unwrap_or(0),
        })
    }
}

/// The tombstones of `md`, oldest first.
pub fn tombstones(md: &Metadata) -> Vec<Tombstone> {
    md.customs.get(CUSTOM).and_then(|g| g["tombstones"].as_array()).map(|a| a.iter().filter_map(Tombstone::from_json).collect()).unwrap_or_default()
}

/// Whether a tombstone of `md` names index uuid `uuid`.
pub fn contains(md: &Metadata, uuid: &str) -> bool {
    md.customs.get(CUSTOM).and_then(|g| g["tombstones"].as_array()).is_some_and(|a| a.iter().any(|t| t["index"]["index_uuid"].as_str() == Some(uuid)))
}

/// Write `stones` (oldest first) into `md`, keeping the newest `max`.
fn store(md: &mut Metadata, mut stones: Vec<Tombstone>, max: usize) {
    if stones.len() > max {
        stones.drain(..stones.len() - max);
    }
    md.customs.insert(CUSTOM.into(), json!({"tombstones": stones.iter().map(Tombstone::to_json).collect::<Vec<_>>()}));
}

/// The graveyard of `md` with a tombstone for index `name`/`uuid` added
/// (ES `TransportDeleteDanglingIndexAction`), keeping the newest `max`.
pub fn with_tombstone(md: &Metadata, name: &str, uuid: &str, now_millis: i64, max: usize) -> Value {
    let mut stones = tombstones(md);
    if !stones.iter().any(|t| t.index_uuid == uuid) {
        stones.push(Tombstone { index_name: name.to_string(), index_uuid: uuid.to_string(), delete_date_in_millis: now_millis });
    }
    let mut out = Metadata::default();
    store(&mut out, stones, max.max(1));
    out.customs.remove(CUSTOM).unwrap_or_default()
}

/// ES `MetadataDeleteIndexService` / `IndexGraveyard.Builder`: every index
/// incarnation of `base` that `new` no longer holds (by uuid: a re-created
/// name is a deletion of the old incarnation) gets a tombstone, dated
/// `now_millis`. Tombstones are never dropped but by the size bound: those
/// of `base` are kept even if a task built `new`'s customs from an older
/// metadata. Returns whether anything was added.
pub fn record_deletions(base: &Metadata, new: &mut Metadata, now_millis: i64, max: usize) -> bool {
    let live: BTreeSet<&str> = new.indices.values().map(|im| im.uuid.as_str()).collect();
    let deleted: Vec<Tombstone> = base.indices.values().filter(|im| !live.contains(im.uuid.as_str()))
        .map(|im| Tombstone { index_name: im.name.clone(), index_uuid: im.uuid.clone(), delete_date_in_millis: now_millis }).collect();
    let mut stones = tombstones(base);
    let mut changed = false;
    for t in tombstones(new) {
        if !stones.iter().any(|s| s.index_uuid == t.index_uuid) {
            stones.push(t);
            changed = true;
        }
    }
    for t in deleted {
        if !stones.iter().any(|s| s.index_uuid == t.index_uuid) {
            stones.push(t);
            changed = true;
        }
    }
    let had = new.customs.get(CUSTOM).is_some() || base.customs.get(CUSTOM).is_some();
    if changed || (had && new.customs.get(CUSTOM) != base.customs.get(CUSTOM)) {
        store(new, stones, max.max(1));
        return changed;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::IndexMetadata;

    fn md(indices: &[(&str, &str)]) -> Metadata {
        let mut m = Metadata::default();
        for (name, uuid) in indices {
            let im = IndexMetadata { name: name.to_string(), uuid: uuid.to_string(), version: 1, mapping_version: 1, settings_version: 1, aliases_version: 1,
                state: crate::state::IndexState::Open, settings: serde_json::Map::new().into(), mappings: json!({}).into(), aliases: Default::default(),
                number_of_shards: 1, number_of_replicas: 0, routing_num_shards: 1, primary_terms: vec![1], in_sync_allocation_ids: Default::default(),
                creation_date: 0, lifecycle: None };
            m.indices.insert(name.to_string(), im);
        }
        m
    }

    #[test]
    fn deletions_and_recreations_are_tombstoned_and_bounded() {
        let base = md(&[("a", "ua"), ("b", "ub"), ("c", "uc")]);
        // `a` deleted, `b` re-created under a new uuid, `c` unchanged.
        let mut new = md(&[("b", "ub2"), ("c", "uc")]);
        assert!(record_deletions(&base, &mut new, 7, DEFAULT_SIZE));
        let t = tombstones(&new);
        assert_eq!(t.iter().map(|t| t.index_uuid.as_str()).collect::<Vec<_>>(), vec!["ua", "ub"]);
        assert_eq!(t[0], Tombstone { index_name: "a".into(), index_uuid: "ua".into(), delete_date_in_millis: 7 });
        assert!(contains(&new, "ua") && contains(&new, "ub") && !contains(&new, "uc") && !contains(&new, "ub2"));
        assert_eq!(new.customs[CUSTOM]["tombstones"][0], json!({"index": {"index_name": "a", "index_uuid": "ua"}, "delete_date_in_millis": 7}));
        // A task that rebuilt the customs from older metadata loses none.
        let base2 = new.clone();
        let mut newer = md(&[("c", "uc")]);
        assert!(record_deletions(&base2, &mut newer, 8, DEFAULT_SIZE));
        assert_eq!(tombstones(&newer).len(), 3);
        // Nothing deleted: nothing changes.
        let mut same = newer.clone();
        assert!(!record_deletions(&newer, &mut same, 9, DEFAULT_SIZE));
        assert_eq!(same, newer);
        // The newest `max` are kept.
        let many: Vec<(String, String)> = (0..10).map(|i| (format!("i{i}"), format!("u{i}"))).collect();
        let base = md(&many.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect::<Vec<_>>());
        let mut empty = Metadata::default();
        record_deletions(&base, &mut empty, 1, 4);
        assert_eq!(tombstones(&empty).iter().map(|t| t.index_uuid.clone()).collect::<Vec<_>>(), vec!["u6", "u7", "u8", "u9"]);
        // An explicit tombstone (a deleted dangling index), bounded too.
        let g = with_tombstone(&empty, "d", "ud", 3, 4);
        let mut m = Metadata::default();
        m.customs.insert(CUSTOM.into(), g);
        assert_eq!(tombstones(&m).iter().map(|t| t.index_uuid.clone()).collect::<Vec<_>>(), vec!["u7", "u8", "u9", "ud"]);
        assert_eq!(with_tombstone(&m, "d", "ud", 4, 4), m.customs[CUSTOM], "idempotent");
    }
}
