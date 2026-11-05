//! Which text fields have had fielddata loaded on this node (ES loads it on
//! the shard copies a sort or aggregation actually runs on). `_cat/fielddata`
//! lists only loaded fields, so on a cluster only the nodes that served such
//! a search report them.

use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet};

static LOADED: Mutex<BTreeMap<String, BTreeSet<String>>> = Mutex::new(BTreeMap::new());

/// A sort or aggregation on `index`/`field` loaded its fielddata.
pub fn mark(index: &str, field: &str) {
    let mut m = LOADED.lock();
    if !m.get(index).is_some_and(|s| s.contains(field)) {
        m.entry(index.to_string()).or_default().insert(field.to_string());
    }
}

pub fn loaded(index: &str, field: &str) -> bool {
    LOADED.lock().get(index).is_some_and(|s| s.contains(field))
}

/// A new index of this name starts with nothing loaded.
pub fn clear_index(index: &str) {
    LOADED.lock().remove(index);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_until_index_recreated() {
        assert!(!loaded("fdc-test", "foo"));
        mark("fdc-test", "foo");
        assert!(loaded("fdc-test", "foo"));
        assert!(!loaded("fdc-test", "bar"));
        clear_index("fdc-test");
        assert!(!loaded("fdc-test", "foo"));
    }
}
