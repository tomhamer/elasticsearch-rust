//! Per-shard engines for multi-shard indices.
//!
//! The engine's `IndexService` can hold several shards, but the cluster
//! places shard copies independently, so a node keeps one engine per shard
//! number: engine 0 is the node's main engine
//! (`ClusterNode::engine`, which also serves every cluster-level API), and
//! shard `k >= 1` of every index with more than `k` shards lives in
//! `data_path/engine-shard-<k>`. Shard `k` of index `i` is always the index
//! named `i` in engine `k`, so a shard copy is addressed by
//! `(engine_for(k), index)` everywhere replication and recovery used
//! `(engine, index)` before.
//!
//! The front end selects the engine for a request with the internal
//! [`SHARD_HEADER`]; it is stripped from client requests.

use crate::mirror::{self, ApplyReport, UuidMap};
use crate::node::ClusterNode;
use elasticsearch_cluster::state::Metadata;
use elasticsearch_engine::node::Node as Engine;
use parking_lot::{Mutex, RwLock};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Internal request header naming the shard a request runs against.
pub const SHARD_HEADER: &str = "x-es-cluster-shard";
pub const PREFILTER_SHARDS_HEADER: &str = "x-es-cluster-prefilter-shards";

pub struct ShardEngine {
    pub engine: Arc<Engine>,
    pub router: axum::Router,
    pub uuids: Mutex<UuidMap>,
}

pub struct ShardEngines {
    data_path: PathBuf,
    primary_engine: Arc<Engine>,
    /// Node settings copied into every shard engine (as engine 0).
    node_settings: serde_json::Map<String, Value>,
    map: RwLock<BTreeMap<u32, Arc<ShardEngine>>>,
    /// The node's dynamic-mapping hook, installed on every shard engine.
    mapping_hook: RwLock<Option<Arc<dyn elasticsearch_engine::index::MappingUpdateHook>>>,
}

fn engine_dir(data_path: &Path, k: u32) -> PathBuf {
    data_path.join(format!("engine-shard-{k}"))
}

fn uuids_path(data_path: &Path, k: u32) -> PathBuf {
    data_path.join(format!("_cluster/uuids-shard-{k}.json"))
}

impl ShardEngines {
    /// Open the shard engines already on disk.
    pub fn open(data_path: &Path, node_settings: serde_json::Map<String, Value>, primary_engine: Arc<Engine>) -> ShardEngines {
        let se = ShardEngines { data_path: data_path.to_path_buf(), primary_engine, node_settings, map: RwLock::new(BTreeMap::new()), mapping_hook: RwLock::new(None) };
        if let Ok(rd) = std::fs::read_dir(data_path) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if let Some(k) = name.strip_prefix("engine-shard-").and_then(|k| k.parse::<u32>().ok()) {
                    if k >= 1 {
                        let _ = se.get_or_open(k);
                    }
                }
            }
        }
        se
    }

    /// The engine of shard `k` (k >= 1) if it is open.
    pub fn get(&self, k: u32) -> Option<Arc<ShardEngine>> {
        self.map.read().get(&k).cloned()
    }

    pub fn get_or_open(&self, k: u32) -> Result<Arc<ShardEngine>, String> {
        if let Some(e) = self.get(k) {
            return Ok(e);
        }
        let mut m = self.map.write();
        if let Some(e) = m.get(&k) {
            return Ok(e.clone());
        }
        let engine = Engine::open_with_security_named(&engine_dir(&self.data_path, k), self.primary_engine.security.clone(), &self.primary_engine.name).map_err(|e| e.to_string())?;
        *engine.transport_address.write() = self.primary_engine.transport_address.read().clone();
        *engine.cluster_transport_addresses.write() = self.primary_engine.cluster_transport_addresses.read().clone();
        *engine.node_settings.write() = self.node_settings.clone();
        if let Some(hook) = self.mapping_hook.read().clone() {
            engine.set_mapping_update_hook(Some(hook));
        }
        // query_then_fetch contexts are capped per node, over every engine.
        *engine.fetch_budget.lock() = self.primary_engine.fetch_budget.lock().clone();
        let router = elasticsearch_http::router(engine.clone());
        let uuids = UuidMap::load(&uuids_path(&self.data_path, k));
        let se = Arc::new(ShardEngine { engine, router, uuids: Mutex::new(uuids) });
        m.insert(k, se.clone());
        Ok(se)
    }

    /// Install the node's dynamic-mapping hook on every shard engine, open
    /// now or later.
    pub fn set_mapping_update_hook(&self, hook: Arc<dyn elasticsearch_engine::index::MappingUpdateHook>) {
        *self.mapping_hook.write() = Some(hook.clone());
        for (_, se) in self.all() {
            se.engine.set_mapping_update_hook(Some(hook.clone()));
        }
    }

    pub(crate) fn admitted_ids(&self)->Result<crate::shared_sequence::SharedSequence<u32>,elasticsearch_engine::error::EsError> {
        crate::shared_sequence::SharedSequence::try_collect(self.map.read().keys().copied())
    }

    /// Open shard engines, ascending.
    pub fn all(&self) -> Vec<(u32, Arc<ShardEngine>)> {
        self.map.read().iter().map(|(k, v)| (*k, v.clone())).collect()
    }
}

/// The metadata shard engine `k` mirrors: the indices with more than `k`
/// shards, no data streams (a multi-shard backing index is written by its
/// concrete name), and auto-create off so a misrouted request can never
/// create an index outside the cluster state.
pub fn shard_metadata(md: &Metadata, k: u32) -> Metadata {
    let mut out = md.clone();
    out.indices.retain(|_, im| im.number_of_shards > k);
    out.customs.remove(mirror::DATA_STREAMS_CUSTOM);
    out.index_templates.clear();
    out.component_templates.clear();
    out.legacy_templates.clear();
    out.transient_settings.remove("action.auto_create_index");
    out.persistent_settings.insert("action.auto_create_index".into(), Value::String("false".into()));
    out
}

/// Highest shard count among the indices.
pub fn max_shards(md: &Metadata) -> u32 {
    md.indices.values().map(|im| im.number_of_shards).max().unwrap_or(1).max(1)
}

/// The document id a single-document request addresses
/// (`/{index}/{_doc|_create|_update|_source|_explain|_termvectors}/{id}`).
pub fn doc_id_of(segments: &[String]) -> Option<&str> {
    match segments {
        [_, kind, id] if matches!(kind.as_str(), "_doc" | "_create" | "_update" | "_source" | "_explain" | "_termvectors") => Some(id.as_str()),
        _ => None,
    }
}

/// The shard a document of `index` is routed to (murmur3 over `_id` or
/// `routing`, as ES `IndexRouting`). An index absent from `md` is shard 0.
pub fn doc_shard(md: &Metadata, index: &str, id: &str, routing: Option<&str>) -> Result<u32, elasticsearch_cluster::routing::RoutingError> {
    // Time series: the `_id` carries the routing hash of its dimensions.
    if let Some(im) = md.indices.get(index).filter(|im| im.number_of_shards > 1 && is_time_series(md, index)) {
        if let Some(h) = elasticsearch_engine::tsdb::routing_hash_of_id(id) {
            return Ok(elasticsearch_cluster::routing::index_routing(im).hash_to_shard(h));
        }
    }
    match md.indices.get(index) {
        Some(im) if im.number_of_shards > 1 => elasticsearch_cluster::routing::index_routing(im).shard_id(id, routing),
        _ => Ok(0),
    }
}

/// ES rejects, before routing, document operations a time series index
/// cannot serve (`IndexRouting.ExtractFromSource`): updates, and any
/// operation that names a routing. `op` is `index`, `create`, `delete`,
/// `update` or `get`.
pub fn tsdb_doc_error(md: &Metadata, index: &str, op: &str, routing: Option<&str>) -> Option<elasticsearch_engine::error::EsError> {
    if !is_time_series(md, index) {
        return None;
    }
    if op == "update" {
        return Some(elasticsearch_engine::tsdb::mode_error("update", index));
    }
    routing.map(|_| elasticsearch_engine::tsdb::mode_error("specifying routing", index))
}

/// `index.mode: time_series`.
pub fn is_time_series(md: &Metadata, index: &str) -> bool {
    md.indices.get(index).is_some_and(|im| {
        let m = im.settings.get("index.mode").and_then(|v| v.as_str()).map(str::to_ascii_lowercase);
        m.as_deref() == Some("time_series")
    })
}

/// The shard an id-less write of `source` into time series `index` goes
/// to: its routing-path dimensions hashed (`IndexRouting.ExtractFromSource`);
/// the primary then generates the matching `_id`.
pub fn tsdb_source_shard(md: &Metadata, index: &str, source: &serde_json::Value) -> u32 {
    let Some(im) = md.indices.get(index).filter(|im| im.number_of_shards > 1) else { return 0 };
    let flat: elasticsearch_engine::settings::FlatSettings = im.settings.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    match elasticsearch_engine::tsdb::source_routing_hash(&im.mappings, &flat, source) {
        Ok(h) => elasticsearch_cluster::routing::index_routing(im).hash_to_shard(h),
        // The primary reports the document's error.
        Err(_) => 0,
    }
}

/// Request-local TSDB routing preparation. Metadata version changes invalidate
/// the parsed mapping; no cross-request cache or shared locks are introduced.
#[derive(Default)]
pub struct TsdbRoutingCache {
    entries: std::collections::HashMap<String, TsdbRoutingEntry>,
}
struct TsdbRoutingEntry {
    key: (String, u64, u64, u64),
    mapping: Option<(elasticsearch_engine::mapping::Mappings, elasticsearch_engine::tsdb::PreparedFields)>,
    settings: elasticsearch_engine::settings::FlatSettings,
}
impl TsdbRoutingCache {
    pub fn source_shard(&mut self, md: &Metadata, index: &str, source: &Value) -> u32 {
        let Some(im) = md.indices.get(index).filter(|im| im.number_of_shards > 1) else { return 0 };
        let key = (im.uuid.clone(), im.version, im.mapping_version, im.settings_version);
        if self.entries.get(index).is_none_or(|e| e.key != key) {
            self.entries.insert(index.into(), TsdbRoutingEntry {
                key,
                mapping: elasticsearch_engine::mapping::Mappings::parse(&im.mappings,
                    &elasticsearch_engine::analysis::AnalysisRegistry::default()).ok().map(|m| {
                        let fields = elasticsearch_engine::tsdb::PreparedFields::new(&m);
                        (m, fields)
                    }),
                settings: im.settings.iter().map(|(k,v)| (k.clone(),v.clone())).collect(),
            });
        }
        let entry = &self.entries[index];
        let Some((mapping, fields)) = &entry.mapping else { return 0 };
        fields.build_ids(mapping, &entry.settings, source).ok()
            .and_then(|ids| elasticsearch_engine::tsdb::routing_hash_of_id(&ids.id))
            .map_or(0, |hash| elasticsearch_cluster::routing::index_routing(im).hash_to_shard(hash))
    }
}

/// The shards a search/count of `index` covers: all of them, or only those
/// the comma-separated `routing` values hash to (ES `searchShards`).
pub fn search_shards(md: &Metadata, index: &str, routing: Option<&str>) -> Vec<u32> {
    let n = shards_of(md, index);
    match (md.indices.get(index), routing.filter(|r| !r.is_empty())) {
        (Some(im), Some(r)) if n > 1 => {
            let ir = elasticsearch_cluster::routing::index_routing(im);
            let set: std::collections::BTreeSet<u32> = r.split(',').flat_map(|v| ir.search_shards(v.trim())).collect();
            set.into_iter().collect()
        }
        _ => (0..n).collect(),
    }
}

/// Shard count of `index` (1 when unknown).
pub fn shards_of(md: &Metadata, index: &str) -> u32 {
    md.indices.get(index).map(|im| im.number_of_shards.max(1)).unwrap_or(1)
}

/// The ES error for a write/get without routing on a partitioned index.
pub fn routing_missing(index: &str, id: &str) -> crate::front::RestResp {
    crate::front::RestResp::error(400, "routing_missing_exception", &format!("routing is required for [{index}]/[{id}]"))
}

/// Parse the shard header of a request.
pub fn shard_of(headers: &[(String, String)]) -> Option<u32> {
    headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(SHARD_HEADER)).and_then(|(_, v)| v.trim().parse().ok())
}

pub fn strip_header(headers: &mut Vec<(String, String)>) {
    headers.retain(|(k, _)| !k.eq_ignore_ascii_case(SHARD_HEADER));
}

pub fn set_header(headers: &mut Vec<(String, String)>, shard: u32) {
    strip_header(headers);
    headers.push((SHARD_HEADER.to_string(), shard.to_string()));
}

impl ClusterNode {
    /// The engine holding shard `k` of every index on this node.
    pub fn engine_for(&self, k: u32) -> Option<Arc<Engine>> {
        if k == 0 {
            return Some(self.engine.clone());
        }
        self.shard_engines.get(k).map(|s| s.engine.clone())
    }

    pub fn router_for(&self, k: u32) -> Option<axum::Router> {
        if k == 0 {
            return Some(self.router.clone());
        }
        self.shard_engines.get(k).map(|s| s.router.clone())
    }

    /// Run `f` on the uuid map of shard engine `k`.
    pub fn with_uuids<R>(&self, k: u32, f: impl FnOnce(&mut UuidMap) -> R) -> Option<R> {
        if k == 0 {
            return Some(f(&mut self.uuids.lock()));
        }
        let se = self.shard_engines.get(k)?;
        let mut g = se.uuids.lock();
        Some(f(&mut g))
    }

    /// Borrow engines without allocating a parallel Arc table. The callback
    /// must not open/remove engines or mutate either engine registry.
    pub(crate) fn try_visit_shard_engines<E>(&self,mut visit:impl FnMut(u32,&Arc<Engine>)->Result<(),E>)->Result<(),E>{
        visit(0,&self.engine)?;
        for (key,shard) in self.shard_engines.map.read().iter(){visit(*key,&shard.engine)?;}Ok(())
    }

    /// Make every local shard's translog appends durable (graceful
    /// shutdown: an `async`-durability write accepted before SIGTERM is not
    /// lost, as ES's close syncs and flushes its shards). Best effort.
    pub fn sync_translogs_for_shutdown(&self) {
        for (k, engine) in self.all_shard_engines() {
            for svc in engine.all_indices() {
                if let Err(e) = svc.sync_translog() {
                    tracing::warn!(index = %svc.name, shard_engine = k, "translog sync at shutdown failed: {}", e.reason);
                }
            }
        }
    }

    /// Every (shard, engine) pair on this node, engine 0 first.
    pub fn all_shard_engines(&self) -> Vec<(u32, Arc<Engine>)> {
        let mut v = vec![(0, self.engine.clone())];
        v.extend(self.shard_engines.all().into_iter().map(|(k, s)| (k, s.engine.clone())));
        v
    }

    /// Mirror `md` into the shard engines (engine 0 is mirrored by the
    /// applier itself). Opens the engines new indices need.
    pub fn apply_shard_engines(&self, prev: Option<&Metadata>, md: &Metadata) -> Vec<(u32, ApplyReport)> {
        let max = max_shards(md);
        let mut out = Vec::new();
        for k in 1..max {
            match self.shard_engines.get_or_open(k) {
                Ok(_) => {}
                Err(e) => {
                    tracing::error!(shard = k, "failed to open shard engine: {e}");
                }
            }
        }
        for (k, se) in self.shard_engines.all() {
            let smd = shard_metadata(md, k);
            // Shard engines never own endpoints: they serve the master's
            // (applied before the indices whose copies resolve them).
            let scripts = crate::meta_extra::apply_stored_scripts(&se.engine, md);
            let inference = crate::meta_extra::apply_inference(&se.engine, md);
            // Incremental when the engine already mirrors `prev` (an engine
            // opened by this application starts with a full pass).
            let mut rep = match prev.filter(|p| max_shards(p) > k) {
                Some(p) => mirror::apply_changes(&se.engine, &shard_metadata(p, k), &smd, &mut se.uuids.lock(), true, &Default::default()),
                None => mirror::apply(&se.engine, &smd, &mut se.uuids.lock()),
            };
            if let Some(e) = scripts {
                rep.errors.push(("stored scripts".into(), e));
            }
            if let Some(e) = inference {
                rep.errors.push(("inference endpoints".into(), e));
            }
            out.push((k, rep));
        }
        out
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use elasticsearch_cluster::state::IndexMetadata;

    fn im(n: u32) -> IndexMetadata {
        let mut m: IndexMetadata = serde_json::from_value(serde_json::json!({
            "name": "x", "uuid": "u", "version": 1, "mapping_version": 1, "settings_version": 1,
            "aliases_version": 1, "state": "open", "settings": {}, "mappings": {}, "aliases": {},
            "number_of_shards": 1, "number_of_replicas": 0, "routing_num_shards": 1,
            "primary_terms": [1], "in_sync_allocation_ids": {}, "creation_date": 0
        }))
        .unwrap_or_else(|_| panic!("index metadata shape"));
        m.number_of_shards = n;
        m
    }

    #[test]
    fn cached_tsdb_routing_matches_uncached_and_refreshes_versions() {
        let mut md = Metadata::default();
        let mut ts = im(4);
        ts.routing_num_shards = 4;
        ts.settings.insert("index.mode".into(), serde_json::json!("time_series"));
        ts.settings.insert("index.routing_path".into(), serde_json::json!(["host"]));
        ts.mappings = serde_json::json!({"properties":{
            "@timestamp":{"type":"date"},
            "host":{"type":"keyword","time_series_dimension":true},
            "rack":{"type":"long","time_series_dimension":true}
        }}).into();
        md.indices.insert("ts".into(), ts);
        let mut cache = TsdbRoutingCache::default();
        let mut seen = std::collections::BTreeSet::new();
        for version in 0..3 {
            for n in 0..64 {
                let doc = serde_json::json!({"@timestamp":"2026-01-01T00:00:00Z","host":format!("host-{n}"),"rack":n});
                let expected = tsdb_source_shard(&md,"ts",&doc);
                assert_eq!(cache.source_shard(&md,"ts",&doc),expected);
                seen.insert(expected);
            }
            let ts = md.indices.get_mut("ts").unwrap();
            if version == 0 {
                ts.settings.insert("index.routing_path".into(),serde_json::json!(["rack"]));
                ts.settings_version += 1;
            } else {
                ts.mappings["properties"]["host"]["time_series_dimension"] = serde_json::json!(false);
                ts.mapping_version += 1;
            }
        }
        assert_eq!(seen.len(),4,"routing must exercise real hashes, not error fallback");
        assert_eq!(cache.source_shard(&md,"ts",&serde_json::json!({})),0);
        assert_eq!(cache.source_shard(&md,"missing",&serde_json::json!({})),0);
    }

    #[test]
    fn shard_metadata_keeps_indices_with_that_shard() {
        let mut md = Metadata::default();
        md.indices.insert("one".into(), im(1));
        md.indices.insert("three".into(), im(3));
        md.customs.insert(mirror::DATA_STREAMS_CUSTOM.into(), serde_json::json!({}));
        md.persistent_settings.insert("action.auto_create_index".into(), "true".into());
        let s1 = shard_metadata(&md, 1);
        assert_eq!(s1.indices.keys().collect::<Vec<_>>(), vec!["three"]);
        assert!(!s1.customs.contains_key(mirror::DATA_STREAMS_CUSTOM));
        assert_eq!(s1.persistent_settings["action.auto_create_index"], "false");
        assert!(shard_metadata(&md, 3).indices.is_empty());
        assert_eq!(max_shards(&md), 3);
        assert_eq!(max_shards(&Metadata::default()), 1);
    }

    #[test]
    fn extract_keeps_number_of_shards() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = Engine::open(&tmp.path().join("e")).unwrap();
        engine.create_index("three", &serde_json::json!({"settings": {"number_of_shards": 3, "number_of_replicas": 1}})).unwrap();
        engine.create_index("one", &serde_json::json!({})).unwrap();
        let mut uuids = UuidMap::load(&tmp.path().join("u.json"));
        let md = mirror::extract(&engine, &Metadata::default(), &mut uuids);
        let three = &md.indices["three"];
        assert_eq!(three.number_of_shards, 3);
        assert_eq!(three.number_of_replicas, 1);
        assert_eq!(three.primary_terms.len(), 3);
        assert_eq!(md.indices["one"].number_of_shards, 1);
        assert_eq!(shards_of(&md, "three"), 3);
        assert_eq!(shards_of(&md, "missing"), 1);
    }

    #[test]
    fn time_series_rejects_updates_and_routing_before_routing() {
        let mut md = Metadata::default();
        let mut ts = im(2);
        ts.settings.insert("index.mode".into(), serde_json::json!("time_series"));
        md.indices.insert("ts".into(), ts);
        md.indices.insert("plain".into(), im(2));
        let reason = |e: Option<elasticsearch_engine::error::EsError>| e.map(|e| e.to_json_with_trace(false)["error"]["reason"].as_str().unwrap_or("").to_string());
        assert_eq!(reason(tsdb_doc_error(&md, "ts", "update", None)).as_deref(), Some("update is not supported because the destination index [ts] is in time_series mode"));
        assert_eq!(reason(tsdb_doc_error(&md, "ts", "get", Some("r"))).as_deref(), Some("specifying routing is not supported because the destination index [ts] is in time_series mode"));
        assert!(tsdb_doc_error(&md, "ts", "index", None).is_none());
        assert!(tsdb_doc_error(&md, "plain", "update", Some("r")).is_none());
    }

    #[test]
    fn doc_shard_routes_by_id_and_routing() {
        let mut md = Metadata::default();
        let mut m3 = im(3);
        m3.routing_num_shards = elasticsearch_cluster::routing::calculate_num_routing_shards(3);
        m3.primary_terms = vec![1; 3];
        md.indices.insert("three".into(), m3);
        md.indices.insert("one".into(), im(1));
        let mut seen = [0u32; 3];
        for i in 0..300 {
            let s = doc_shard(&md, "three", &format!("{i}"), None).unwrap();
            assert!(s < 3);
            seen[s as usize] += 1;
            // routing overrides the id
            assert_eq!(doc_shard(&md, "three", &format!("{i}"), Some("r")).unwrap(), doc_shard(&md, "three", "x", Some("r")).unwrap());
        }
        assert!(seen.iter().all(|&n| n > 50), "{seen:?}");
        assert_eq!(search_shards(&md, "three", None), vec![0, 1, 2]);
        assert_eq!(search_shards(&md, "three", Some("r")), vec![doc_shard(&md, "three", "x", Some("r")).unwrap()]);
        assert_eq!(search_shards(&md, "one", Some("r")), vec![0]);
        assert_eq!(doc_shard(&md, "one", "a", None).unwrap(), 0);
        assert_eq!(doc_shard(&md, "missing", "a", None).unwrap(), 0);
        let segs: Vec<String> = ["i", "_doc", "7"].iter().map(|s| s.to_string()).collect();
        assert_eq!(doc_id_of(&segs), Some("7"));
        let segs: Vec<String> = ["i", "_search"].iter().map(|s| s.to_string()).collect();
        assert_eq!(doc_id_of(&segs), None);
    }

    #[test]
    fn shard_header_round_trip() {
        let mut h = vec![("X-ES-Cluster-Shard".to_string(), "9".to_string()), ("a".into(), "b".into())];
        assert_eq!(shard_of(&h), Some(9));
        set_header(&mut h, 2);
        assert_eq!(shard_of(&h), Some(2));
        assert_eq!(h.len(), 2);
        strip_header(&mut h);
        assert_eq!(shard_of(&h), None);
    }
}
