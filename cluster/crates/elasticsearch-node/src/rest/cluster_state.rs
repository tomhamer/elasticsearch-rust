//! `GET /_cluster/state[/{metrics}[/{indices}]]` in Elasticsearch's shape.

use elasticsearch_cluster::state::{ClusterBlock, ClusterState, ShardRouting, ShardState, UnassignedInfo};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

/// The metrics ES accepts; `_all` expands to all of them.
pub const METRICS: &[&str] = &["version", "master_node", "blocks", "nodes", "metadata", "routing_table", "routing_nodes", "customs"];

pub fn parse_metrics(spec: Option<&str>) -> Result<BTreeSet<String>, String> {
    let Some(spec) = spec.filter(|s| !s.is_empty()) else {
        return Ok(METRICS.iter().map(|s| s.to_string()).collect());
    };
    let mut out = BTreeSet::new();
    for m in spec.split(',') {
        if m == "_all" {
            out.extend(METRICS.iter().map(|s| s.to_string()));
        } else if METRICS.contains(&m) {
            out.insert(m.to_string());
        } else {
            return Err(format!("request [/_cluster/state/{spec}] contains unrecognized metric: [{m}]"));
        }
    }
    Ok(out)
}

/// ES renders millisecond timestamps in unassigned_info as ISO-8601 UTC.
pub fn iso_millis(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms).map(|d| d.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()).unwrap_or_default()
}

/// Flat `a.b.c` settings → nested objects (ES's default rendering).
pub fn nest_settings(flat: &Map<String, Value>) -> Value {
    let mut root = Map::new();
    for (k, v) in flat {
        let parts: Vec<&str> = k.split('.').collect();
        let mut cur = &mut root;
        for p in &parts[..parts.len() - 1] {
            let e = cur.entry(p.to_string()).or_insert_with(|| Value::Object(Map::new()));
            if !e.is_object() {
                *e = Value::Object(Map::new());
            }
            cur = e.as_object_mut().expect("object");
        }
        cur.insert(parts[parts.len() - 1].to_string(), v.clone());
    }
    Value::Object(root)
}

fn block_json(b: &ClusterBlock) -> Value {
    let mut m = Map::new();
    m.insert("description".into(), json!(b.description));
    m.insert("retryable".into(), json!(b.retryable));
    if b.disable_state_persistence {
        m.insert("disable_state_persistence".into(), json!(true));
    }
    m.insert("levels".into(), json!(b.levels));
    Value::Object(m)
}

fn derived_index_blocks(im: &elasticsearch_cluster::state::IndexMetadata) -> Vec<(u32, Value)> {
    let on = |k: &str| matches!(im.settings.get(k), Some(Value::String(s)) if s == "true") || im.settings.get(k) == Some(&Value::Bool(true));
    let block = |desc: &str, retryable: bool, levels: &[&str]| json!({"description": desc, "retryable": retryable, "levels": levels});
    let mut out = Vec::new();
    if im.state == elasticsearch_cluster::state::IndexState::Close {
        out.push((4, block("index closed", false, &["read", "write"])));
    }
    if on("index.blocks.read_only") {
        out.push((5, block("index read-only (api)", false, &["write", "metadata_write"])));
    }
    if on("index.blocks.read") {
        out.push((7, block("index read (api)", false, &["read"])));
    }
    if on("index.blocks.write") {
        out.push((8, block("index write (api)", false, &["write"])));
    }
    if on("index.blocks.metadata") {
        out.push((9, block("index metadata (api)", false, &["metadata_read", "metadata_write"])));
    }
    if on("index.blocks.read_only_allow_delete") {
        out.push((12, block("disk usage exceeded flood-stage watermark, index has read-only-allow-delete block; for more information, see https://www.elastic.co/guide/en/elasticsearch/reference/8.19/fix-watermark-errors.html", false, &["write", "metadata_write"])));
    }
    out
}

pub fn unassigned_info_json(u: &UnassignedInfo) -> Value {
    let mut m = Map::new();
    m.insert("reason".into(), serde_json::to_value(u.reason).unwrap_or(Value::Null));
    m.insert("at".into(), json!(iso_millis(u.at_millis)));
    if u.failed_allocations > 0 {
        m.insert("failed_attempts".into(), json!(u.failed_allocations));
    }
    m.insert("delayed".into(), json!(u.delayed));
    if let Some(d) = &u.details {
        m.insert("details".into(), json!(d));
    }
    m.insert("allocation_status".into(), json!(u.last_allocation_status));
    Value::Object(m)
}

pub fn shard_routing_json(s: &ShardRouting) -> Value {
    let mut m = Map::new();
    let state = match s.state {
        ShardState::Unassigned => "UNASSIGNED",
        ShardState::Initializing => "INITIALIZING",
        ShardState::Started => "STARTED",
        ShardState::Relocating => "RELOCATING",
    };
    m.insert("state".into(), json!(state));
    m.insert("primary".into(), json!(s.primary));
    m.insert("node".into(), json!(s.node_id));
    m.insert("relocating_node".into(), json!(s.relocating_node_id));
    m.insert("shard".into(), json!(s.shard));
    m.insert("index".into(), json!(s.index));
    if let Some(sz) = s.expected_shard_size_in_bytes {
        if s.state == ShardState::Initializing {
            m.insert("expected_shard_size_in_bytes".into(), json!(sz));
        }
    }
    if let Some(rs) = &s.recovery_source {
        if s.state == ShardState::Initializing || s.state == ShardState::Unassigned {
            m.insert("recovery_source".into(), serde_json::to_value(rs).unwrap_or(Value::Null));
        }
    }
    if let Some(a) = &s.allocation_id {
        let mut am = Map::new();
        am.insert("id".into(), json!(a.id));
        if let Some(r) = &a.relocation_id {
            am.insert("relocation_id".into(), json!(r));
        }
        m.insert("allocation_id".into(), Value::Object(am));
    }
    if let Some(u) = &s.unassigned_info {
        m.insert("unassigned_info".into(), unassigned_info_json(u));
    }
    Value::Object(m)
}

/// Render the state. `indices` restricts metadata / routing_table /
/// blocks to those concrete index names (None = all).
pub fn render(state: &ClusterState, metrics: &BTreeSet<String>, indices: Option<&BTreeSet<String>>) -> Value {
    let want = |k: &str| metrics.contains(k);
    let keep = |name: &str| indices.map(|s| s.contains(name)).unwrap_or(true);
    let mut out = Map::new();
    out.insert("cluster_name".into(), json!(state.cluster_name));
    out.insert("cluster_uuid".into(), json!(state.metadata.cluster_uuid));
    if want("version") {
        out.insert("version".into(), json!(state.version));
        out.insert("state_uuid".into(), json!(state.state_uuid));
    }
    if want("master_node") {
        out.insert("master_node".into(), json!(state.master_node_id));
    }
    if want("blocks") {
        let mut b = Map::new();
        if !state.blocks.global.is_empty() {
            b.insert("global".into(), Value::Object(state.blocks.global.iter().map(|x| (x.id.to_string(), block_json(x))).collect()));
        }
        let mut idx: Map<String, Value> = state
            .blocks
            .indices
            .iter()
            .filter(|(n, s)| keep(n) && !s.is_empty())
            .map(|(n, s)| (n.clone(), Value::Object(s.iter().map(|x| (x.id.to_string(), block_json(x))).collect())))
            .collect();
        // Index blocks ES derives from `index.blocks.*` settings and the
        // closed state (`IndexMetadata` → `ClusterBlocks.updateBlocks`).
        for (name, im) in state.metadata.indices.iter().filter(|(n, _)| keep(n)) {
            for (id, b) in derived_index_blocks(im) {
                let e = idx.entry(name.clone()).or_insert_with(|| Value::Object(Map::new()));
                if let Some(o) = e.as_object_mut() {
                    o.entry(id.to_string()).or_insert(b);
                }
            }
        }
        if !idx.is_empty() {
            b.insert("indices".into(), Value::Object(idx));
        }
        out.insert("blocks".into(), Value::Object(b));
    }
    if want("nodes") {
        let nodes: Map<String, Value> = state
            .nodes
            .values()
            .map(|n| {
                (
                    n.id.clone(),
                    json!({
                        "name": n.name,
                        "ephemeral_id": n.ephemeral_id,
                        "transport_address": n.transport_address,
                        "external_id": n.name,
                        "attributes": n.attributes,
                        "roles": n.roles.iter().map(|r| r.name()).collect::<Vec<_>>(),
                        "version": n.version,
                        "min_index_version": elasticsearch_engine::version::IndexVersion::MINIMUM_COMPATIBLE.id(),
                        "max_index_version": elasticsearch_engine::version::IndexVersion::CURRENT.id(),
                    }),
                )
            })
            .collect();
        out.insert("nodes".into(), Value::Object(nodes));
        out.insert("nodes_features".into(), Value::Array(state.nodes.values().map(|n| elasticsearch_http::node_features(&n.id)).collect()));
    }
    if want("metadata") {
        let md = &state.metadata;
        let c = &state.coordination;
        let mut m = Map::new();
        m.insert("cluster_uuid".into(), json!(md.cluster_uuid));
        m.insert("cluster_uuid_committed".into(), json!(md.cluster_uuid_committed));
        m.insert(
            "cluster_coordination".into(),
            json!({
                "term": c.term,
                "last_committed_config": c.last_committed_config.0,
                "last_accepted_config": c.last_accepted_config.0,
                "voting_config_exclusions": c.voting_config_exclusions.iter().map(|e| json!({"node_id": e.node_id, "node_name": e.node_name})).collect::<Vec<_>>(),
            }),
        );
        m.insert("templates".into(), json!(md.legacy_templates));
        let mut idx = Map::new();
        for (name, im) in md.indices.iter().filter(|(n, _)| keep(n)) {
            let primary_terms: Map<String, Value> = im.primary_terms.iter().enumerate().map(|(i, t)| (i.to_string(), json!(t))).collect();
            let in_sync: Map<String, Value> = (0..im.number_of_shards).map(|s| (s.to_string(), json!(im.in_sync_allocation_ids.get(&s).cloned().unwrap_or_default()))).collect();
            idx.insert(
                name.clone(),
                json!({
                    "version": im.version,
                    "mapping_version": im.mapping_version,
                    "settings_version": im.settings_version,
                    "aliases_version": im.aliases_version,
                    "routing_num_shards": im.routing_num_shards,
                    "state": serde_json::to_value(im.state).unwrap_or(Value::Null),
                    "settings": nest_settings(&im.settings),
                    "mappings": if im.mappings.as_object().map(|o| o.is_empty()).unwrap_or(true) { json!({}) } else { json!({"_doc": im.mappings}) },
                    "aliases": im.aliases.keys().collect::<Vec<_>>(),
                    "primary_terms": primary_terms,
                    "in_sync_allocations": in_sync,
                    "rollover_info": {},
                    "mappings_updated_version": elasticsearch_engine::version::IndexVersion::CURRENT.id(),
                    "system": false,
                    "timestamp_range": {"shards": []},
                }),
            );
        }
        m.insert("indices".into(), Value::Object(idx));
        if !md.index_templates.is_empty() {
            m.insert("index_template".into(), json!({"index_template": md.index_templates}));
        }
        if !md.component_templates.is_empty() {
            m.insert("component_template".into(), json!({"component_template": md.component_templates}));
        }
        // Metadata customs (repositories, ingest, ilm, data streams, ...)
        // live under `metadata`; the `customs` metric is for cluster-state
        // customs (snapshots in progress), rendered by the caller.
        for (k, v) in &md.customs {
            if crate::snapshot_api::is_internal_custom(k) || crate::meta_extra::is_internal_custom(k) || k == elasticsearch_cluster::compatibility::SEGMENT_FORMAT_CUSTOM {
                continue;
            }
            m.insert(k.clone(), v.clone());
        }
        m.entry("index-graveyard").or_insert(json!({"tombstones": []}));
        m.insert("reserved_state".into(), json!({}));
        out.insert("metadata".into(), Value::Object(m));
    }
    if want("routing_table") {
        let mut idx = Map::new();
        for (name, irt) in state.routing_table.indices.iter().filter(|(n, _)| keep(n)) {
            let shards: Map<String, Value> = irt.shards.iter().map(|s| (s.shard.to_string(), Value::Array(s.copies.iter().map(shard_routing_json).collect()))).collect();
            idx.insert(name.clone(), json!({"shards": shards}));
        }
        out.insert("routing_table".into(), json!({"indices": idx}));
    }
    if want("routing_nodes") {
        let all: Vec<&ShardRouting> = state.routing_table.all_shards().filter(|s| keep(&s.index)).collect();
        let unassigned: Vec<Value> = all.iter().filter(|s| s.node_id.is_none()).map(|s| shard_routing_json(s)).collect();
        let mut nodes = Map::new();
        for n in state.data_nodes() {
            nodes.insert(n.id.clone(), Value::Array(all.iter().filter(|s| s.node_id.as_deref() == Some(n.id.as_str())).map(|s| shard_routing_json(s)).collect()));
        }
        out.insert("routing_nodes".into(), json!({"unassigned": unassigned, "nodes": nodes}));
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_parse() {
        assert_eq!(parse_metrics(None).unwrap().len(), METRICS.len());
        assert_eq!(parse_metrics(Some("_all")).unwrap().len(), METRICS.len());
        assert!(parse_metrics(Some("nodes,version")).unwrap().contains("nodes"));
        assert_eq!(parse_metrics(Some("foo")).unwrap_err(), "request [/_cluster/state/foo] contains unrecognized metric: [foo]");
    }

    #[test]
    fn renders_empty_state() {
        let s = ClusterState::empty("c");
        let v = render(&s, &parse_metrics(None).unwrap(), None);
        assert_eq!(v["cluster_name"], "c");
        assert_eq!(v["cluster_uuid"], "_na_");
        assert_eq!(v["blocks"]["global"]["1"]["description"], "state not recovered / initialized");
        assert_eq!(v["metadata"]["cluster_coordination"]["term"], 0);
        let only = render(&s, &parse_metrics(Some("master_node")).unwrap(), None);
        assert!(only.get("nodes").is_none());
        assert!(only.get("master_node").is_some());
    }

    #[test]
    fn iso() {
        assert_eq!(iso_millis(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(nest_settings(&serde_json::from_value(json!({"index.number_of_shards": "1", "index.uuid": "x"})).unwrap())["index"]["number_of_shards"], "1");
    }
}
