//! ES|QL views (`PUT/GET/DELETE /_query/view/{name}`): named queries that
//! `FROM` reads like an index. Views are index abstractions: their names
//! conflict with indices, aliases and data streams.
//!
//! Views are kept per node data path in `_state/esql_views.json`.

use crate::error::{EsError, Result};
use crate::node::Node;
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

pub struct ViewStore {
    path: PathBuf,
    views: Mutex<BTreeMap<String, String>>,
}

fn registry() -> &'static Mutex<HashMap<PathBuf, Arc<ViewStore>>> {
    static R: OnceLock<Mutex<HashMap<PathBuf, Arc<ViewStore>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The view store of a node.
pub fn store(node: &Node) -> Arc<ViewStore> {
    let mut r = registry().lock();
    r.entry(node.data.clone()).or_insert_with(|| Arc::new(ViewStore::open(&node.data))).clone()
}

impl ViewStore {
    fn open(data: &Path) -> ViewStore {
        let path = data.join("_state").join("esql_views.json");
        let mut map = BTreeMap::new();
        if let Ok(b) = std::fs::read(&path) {
            if let Ok(Value::Object(m)) = serde_json::from_slice::<Value>(&b) {
                for (k, v) in m {
                    if let Some(q) = v.as_str() {
                        map.insert(k, q.to_string());
                    }
                }
            }
        }
        ViewStore { path, views: Mutex::new(map) }
    }

    fn persist(&self, map: &BTreeMap<String, String>) -> Result<()> {
        let v: Map<String, Value> = map.iter().map(|(k, q)| (k.clone(), json!(q))).collect();
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&Value::Object(v))?)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<String> {
        self.views.lock().get(name).cloned()
    }

    pub fn names(&self) -> Vec<String> {
        self.views.lock().keys().cloned().collect()
    }

    pub fn all(&self) -> BTreeMap<String, String> {
        self.views.lock().clone()
    }

    /// Every view, as published in cluster state (`name -> query`). ES keeps
    /// views in cluster metadata, so any node can resolve them.
    pub fn cluster_state(&self) -> Value {
        Value::Object(self.views.lock().iter().map(|(k, q)| (k.clone(), json!(q))).collect())
    }

    /// Make this node's views the published ones.
    pub fn restore_cluster_state(&self, v: &Value) -> Result<()> {
        let want: BTreeMap<String, String> = v.as_object().map(|m| m.iter().filter_map(|(k, q)| q.as_str().map(|q| (k.clone(), q.to_string()))).collect()).unwrap_or_default();
        let mut m = self.views.lock();
        if *m != want {
            *m = want;
            self.persist(&m)?;
        }
        Ok(())
    }

    pub fn put(&self, name: &str, query: &str) -> Result<()> {
        let mut m = self.views.lock();
        m.insert(name.to_string(), query.to_string());
        self.persist(&m)
    }

    /// Delete views by name (`_all`, wildcards); every concrete name must exist.
    pub fn delete(&self, names: &[String]) -> Result<()> {
        let mut m = self.views.lock();
        let mut doomed: Vec<String> = Vec::new();
        for n in names {
            if n == "_all" || n == "*" {
                doomed.extend(m.keys().cloned());
            } else if n.contains('*') {
                doomed.extend(m.keys().filter(|k| crate::mapping::wildcard_match(n, k)).cloned());
            } else if m.contains_key(n) {
                doomed.push(n.clone());
            } else {
                return Err(EsError::resource_not_found(format!("view [{n}] not found")));
            }
        }
        for d in doomed {
            m.remove(&d);
        }
        self.persist(&m)
    }
}

/// Whether `name` is a view of this node.
pub fn exists(node: &Node, name: &str) -> bool {
    store(node).get(name).is_some()
}

/// `invalid view name [x], <why>` validation (index-name rules).
pub fn validate_name(name: &str) -> Option<String> {
    let why = if name.is_empty() {
        "must not be empty".to_string()
    } else if name.chars().any(|c| matches!(c, ' ' | '"' | '*' | ',' | '/' | '<' | '>' | '?' | '\\' | '|')) {
        "must not contain the following characters [' ','\"','*',',','/','<','>','?','\\','|']".to_string()
    } else if name.contains('#') {
        "must not contain '#'".to_string()
    } else if name.contains(':') {
        "must not contain ':'".to_string()
    } else if name.starts_with(['_', '-', '+']) {
        "must not start with '_', '-', or '+'".to_string()
    } else if name == "." || name == ".." {
        "must not be '.' or '..'".to_string()
    } else if name.to_lowercase() != name {
        "must be lowercase".to_string()
    } else if name.len() > 255 {
        format!("index name is too long, ({} > 255)", name.len())
    } else {
        return None;
    };
    Some(format!("invalid view name [{name}], {why}"))
}

#[cfg(test)]
mod cluster_state_tests {
    use super::*;

    #[test]
    fn views_round_trip_through_cluster_state() {
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let (sa, sb) = (ViewStore::open(a.path()), ViewStore::open(b.path()));
        sa.put("dogs", "FROM test | WHERE animal == \"dog\"").unwrap();
        sb.restore_cluster_state(&sa.cluster_state()).unwrap();
        assert_eq!(sb.get("dogs").as_deref(), Some("FROM test | WHERE animal == \"dog\""));
        sa.delete(&["dogs".into()]).unwrap();
        sb.restore_cluster_state(&sa.cluster_state()).unwrap();
        assert!(sb.get("dogs").is_none());
        // Survives a reopen (persisted).
        assert!(ViewStore::open(b.path()).get("dogs").is_none());
    }
}
