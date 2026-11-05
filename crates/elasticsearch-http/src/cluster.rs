//! Root info, cluster health/state/settings, nodes, `_xpack/usage`.

use crate::{blocking, Req, Resp, R};
use axum::http::Method;
use elasticsearch_engine::error::EsError;
use elasticsearch_engine::node::{Node, ResolveOpts, ES_VERSION, LUCENE_VERSION, NODE_ID};
use serde_json::{json, Map, Value};
use std::sync::Arc;

pub use elasticsearch_engine::version::{BUILD_DATE, BUILD_HASH};

pub fn root(node: &Node, req: &Req) -> R {
    if req.method != Method::GET && req.method != Method::HEAD {
        return crate::method_not_allowed(req, &["GET", "HEAD"]);
    }
    use elasticsearch_engine::version as v;
    Ok(Resp::ok(json!({
        "name": node.name,
        "cluster_name": node.cluster_name,
        "cluster_uuid": node.cluster_uuid,
        "version": {
            "number": v::ES_VERSION,
            "build_flavor": v::BUILD_FLAVOR,
            "build_type": v::BUILD_TYPE,
            "build_hash": BUILD_HASH,
            "build_date": BUILD_DATE,
            "build_snapshot": false,
            "lucene_version": LUCENE_VERSION,
            "minimum_wire_compatibility_version": v::MIN_WIRE_COMPAT,
            "minimum_index_compatibility_version": v::MIN_INDEX_COMPAT
        },
        "tagline": "You Know, for Search"
    })))
}

pub async fn handle(node: Arc<Node>, req: Req) -> R {
    let a = req.seg(0).to_string();
    let b = req.seg(1).to_string();
    let m = req.method.clone();
    match (a.as_str(), b.as_str(), &m) {
        ("_cluster", "health", &Method::GET) => blocking(move || health(&node, &req)).await,
        ("_cluster", "state", &Method::GET) => blocking(move || state(&node, &req)).await,
        ("_cluster", "settings", &Method::GET) => blocking(move || get_cluster_settings(&node, &req)).await,
        ("_cluster", "settings", &Method::PUT) => blocking(move || put_cluster_settings(&node, &req)).await,
        ("_cluster", "voting_config_exclusions", _) => blocking(move || crate::coordination::handle(&node, &req)).await,
        ("_cluster", "pending_tasks", &Method::GET) => Ok(Resp::ok(json!({"tasks": []}))),
        ("_nodes", _, &Method::GET) => blocking(move || nodes(&node, &req)).await,
        ("_xpack", "usage", &Method::GET) => blocking(move || Ok(Resp::ok(crate::usage::xpack_usage(&node)))).await,
        ("_security", "_authenticate", &Method::GET) => Ok(Resp::ok(json!({
            "username": "_anonymous", "roles": ["superuser"], "full_name": null, "email": null, "metadata": {}, "enabled": true,
            "authentication_realm": {"name": "__anonymous", "type": "__anonymous"}, "lookup_realm": {"name": "__anonymous", "type": "__anonymous"},
            "authentication_type": "anonymous"
        }))),
        ("_tasks", _, &Method::GET) => Ok(Resp::ok(json!({"nodes": {}}))),
        ("_ingest", "pipeline", &Method::GET) => Ok(Resp::ok(json!({}))),
        ("_capabilities", _, &Method::GET) => {
            let list = |k: &str| -> Vec<&str> { req.param(k).map(|v| v.split(',').filter(|s| !s.is_empty()).collect()).unwrap_or_default() };
            let method = req.param("method").unwrap_or("GET");
            let path = req.param("path").unwrap_or("/");
            let supported = crate::features::capability_supported(method, path, &list("parameters"), &list("capabilities")).unwrap_or(Value::Null);
            Ok(Resp::ok(json!({"supported": supported})))
        }
        ("_info", _, &Method::GET) => cluster_info(&node, &req),
        _ => crate::not_found_handler(&req),
    }
}

/// (status, active shards, unassigned shards, unassigned primaries, per-index
/// health) for the indices. A replica is unassigned when the cluster has too
/// few data nodes for it ([`Node::unassigned_replicas`]): its index is
/// yellow. A primary whose allocation failed (an index whose analyzers do not
/// build, [`IndexService::failed_shards`]) makes its index red.
fn health_of_opts(node: &Node, index: Option<&str>, opts: ResolveOpts) -> Result<(String, usize, usize, usize, Vec<(String, Value)>), EsError> {
    let targets = node.resolve(index.unwrap_or("_all"), opts)?;
    let mut active = 0;
    let mut unassigned = 0;
    let mut unassigned_primaries = 0;
    let mut per = Vec::new();
    let mut worst = "green";
    for t in &targets {
        let replicas = t.svc.setting_usize("index.number_of_replicas", 1);
        let shards = t.svc.setting_usize("index.number_of_shards", 1);
        let lost = node.unassigned_replicas(&t.svc);
        let mut failed = t.svc.failed_shards();
        // A copy whose engine failed (tragic event) is not active until it
        // is recovered from its store.
        for sh in 0..shards as u32 {
            if let Some(error) = t.svc.shard(sh).and_then(|c| c.shard.tragic_failure()) {
                failed.entry(sh).or_insert(error);
            }
        }
        let failed_here = failed.len().min(shards);
        let mut idx_active = 0;
        let mut idx_unassigned = 0;
        let mut per_shard = serde_json::Map::new();
        for sh in 0..shards {
            let f = failed.contains_key(&(sh as u32));
            // A shard without its primary has no started replicas either.
            let (sh_active, sh_unassigned) = if f { (0, replicas + 1) } else { (1 + replicas - lost, lost) };
            idx_active += sh_active;
            idx_unassigned += sh_unassigned;
            let st = if f { "red" } else if sh_unassigned > 0 { "yellow" } else { "green" };
            per_shard.insert(
                sh.to_string(),
                json!({"status": st, "primary_active": !f, "active_shards": sh_active, "relocating_shards": 0, "initializing_shards": 0,
                       "unassigned_shards": sh_unassigned, "unassigned_primary_shards": usize::from(f)}),
            );
        }
        let st = if failed_here > 0 { "red" } else if idx_unassigned > 0 { "yellow" } else { "green" };
        if rank(st) > rank(worst) {
            worst = st;
        }
        active += idx_active;
        unassigned += idx_unassigned;
        unassigned_primaries += failed_here;
        per.push((
            t.svc.name.clone(),
            json!({"status": st, "number_of_shards": shards, "number_of_replicas": replicas, "active_primary_shards": shards - failed_here, "active_shards": idx_active,
                   "relocating_shards": 0, "initializing_shards": 0, "unassigned_shards": idx_unassigned, "unassigned_primary_shards": failed_here,
                   "shards": per_shard}),
        ));
    }
    Ok((worst.to_string(), active, unassigned, unassigned_primaries, per))
}

fn rank(s: &str) -> u8 {
    match s {
        "green" => 0,
        "yellow" => 1,
        _ => 2,
    }
}

/// Concrete (non-wildcard) names in a health index expression that do not
/// resolve: ES waits for them to exist.
fn missing_health_targets(node: &Node, index: Option<&str>) -> bool {
    let Some(expr) = index else { return false };
    // The synonyms system index (`.synonyms-2`, alias `.synonyms`) exists
    // once a synonyms set was stored; it is not an engine index here.
    let synonyms_index = |p: &str| matches!(p, ".synonyms" | ".synonyms-2") && !elasticsearch_engine::search_ext::ext(node).synonyms.read().is_empty();
    expr.split(',').map(str::trim).filter(|p| !p.is_empty() && !p.starts_with('-') && !p.contains('*') && *p != "_all" && !synonyms_index(p)).any(|p| {
        let mut o = ResolveOpts::default();
        o.expand_closed = true;
        o.expand_hidden = true;
        node.resolve(p, o).map(|t| t.is_empty()).unwrap_or(true)
    })
}

fn health(node: &Node, req: &Req) -> R {
    let index = req.segs.get(2).cloned();
    // ES default for health: lenient, expand open + closed + hidden.
    let mut o = ResolveOpts::from_params(&req.params);
    if req.param("expand_wildcards").is_none() {
        o.expand_closed = true;
        o.expand_hidden = true;
    }
    if req.param("ignore_unavailable").is_none() {
        o.ignore_unavailable = true;
    }
    if let Some(w) = req.param("wait_for_status") {
        if !matches!(w, "green" | "yellow" | "red") {
            return Err(EsError::illegal_argument(format!("unknown cluster health status [{w}]")));
        }
    }
    let wait_nodes = match req.param("wait_for_nodes") {
        None => None,
        Some(w) => {
            let (op, n) = match w.strip_prefix(">=").or_else(|| w.strip_prefix("ge(").and_then(|r| r.strip_suffix(')'))) {
                Some(n) => (">=", n),
                None => match w.strip_prefix("<=").or_else(|| w.strip_prefix("le(").and_then(|r| r.strip_suffix(')'))) {
                    Some(n) => ("<=", n),
                    None => match w.strip_prefix('>').or_else(|| w.strip_prefix("gt(").and_then(|r| r.strip_suffix(')'))) {
                        Some(n) => (">", n),
                        None => match w.strip_prefix('<').or_else(|| w.strip_prefix("lt(").and_then(|r| r.strip_suffix(')'))) {
                            Some(n) => ("<", n),
                            None => ("==", w),
                        },
                    },
                },
            };
            let n: i64 = n.trim().parse().map_err(|_| EsError::illegal_argument(format!("failed to parse value [{w}] for wait_for_nodes")))?;
            Some((op, n))
        }
    };
    let wait_active = match req.param("wait_for_active_shards") {
        None | Some("0") => None,
        Some("all") => Some(-2i64),
        Some("-1") => None,
        Some(n) => Some(n.parse::<i64>().map_err(|_| EsError::illegal_argument(format!("failed to parse value [{n}] for wait_for_active_shards")))?),
    };
    // `wait_for_events=<priority>` waits for the pending cluster-state tasks
    // of that priority to drain; this node runs none, so it holds at once.
    if let Some(w) = req.param("wait_for_events") {
        if !matches!(w.to_uppercase().as_str(), "IMMEDIATE" | "URGENT" | "HIGH" | "NORMAL" | "LOW" | "LANGUID") {
            return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.common.Priority.{}", w.to_uppercase())));
        }
    }
    // Shards never relocate or initialize here: the conditions hold at once.
    req.flag("wait_for_no_relocating_shards")?;
    req.flag("wait_for_no_initializing_shards")?;
    let data_nodes = node.data_nodes();
    let timeout_ms = match req.param("timeout") {
        None => 30_000,
        Some(t) => elasticsearch_engine::date::parse_time_value(t).ok_or_else(|| EsError::parsing(format!("failed to parse setting [timeout] with value [{t}] as a time value: unit is missing or unrecognized")))?.max(0) as u64,
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
    // ES re-evaluates the health on every cluster state change until the
    // conditions hold or the timeout expires (`408` + `timed_out`).
    let (missing, status, active, unassigned, unassigned_primaries, per) = loop {
        let missing = missing_health_targets(node, index.as_deref());
        let (status, active, unassigned, unassigned_primaries, per) = if missing { ("red".to_string(), 0, 0, 0, Vec::new()) } else { health_of_opts(node, index.as_deref(), o)? };
        let mut ok = !missing;
        if let Some(w) = req.param("wait_for_status") {
            ok &= rank(&status) <= rank(w);
        }
        if let Some((op, n)) = wait_nodes {
            let have = data_nodes as i64;
            ok &= match op {
                ">=" => have >= n,
                "<=" => have <= n,
                ">" => have > n,
                "<" => have < n,
                _ => have == n,
            };
        }
        if let Some(w) = wait_active {
            ok &= if w == -2 { unassigned == 0 } else { active as i64 >= w };
        }
        if ok || std::time::Instant::now() >= deadline {
            break (if ok { None } else { Some(()) }, status, active, unassigned, unassigned_primaries, per);
        }
        std::thread::sleep(std::time::Duration::from_millis(50).min(deadline.saturating_duration_since(std::time::Instant::now())));
    };
    let timed_out = missing.is_some();
    let mut per = per;
    let active_primaries: u64 = per.iter().map(|(_, v)| v["active_primary_shards"].as_u64().unwrap_or(0)).sum();
    let pct = if active + unassigned == 0 { 100.0 } else { active as f64 * 100.0 / (active + unassigned) as f64 };
    let mut body = json!({
        "cluster_name": node.cluster_name,
        "status": status,
        "timed_out": timed_out,
        "number_of_nodes": data_nodes,
        "number_of_data_nodes": data_nodes,
        "active_primary_shards": active_primaries,
        "active_shards": active,
        "relocating_shards": 0,
        "initializing_shards": 0,
        "unassigned_shards": unassigned,
        "unassigned_primary_shards": unassigned_primaries,
        "delayed_unassigned_shards": 0,
        "number_of_pending_tasks": 0,
        "number_of_in_flight_fetch": 0,
        "task_max_waiting_in_queue_millis": 0,
        "active_shards_percent_as_number": pct
    });
    match req.param("level") {
        Some("indices") => {
            for (_, v) in per.iter_mut() {
                if let Some(o) = v.as_object_mut() {
                    o.remove("shards");
                }
            }
            body["indices"] = Value::Object(per.into_iter().collect());
        }
        Some("shards") => body["indices"] = Value::Object(per.into_iter().collect()),
        Some("cluster") | None => {}
        Some(other) => return Err(EsError::illegal_argument(format!("No enum constant org.elasticsearch.cluster.health.ClusterHealthRequest.Level.{}", other.to_uppercase()))),
    }
    Ok(Resp::json(if timed_out { 408 } else { 200 }, body))
}

/// `_nodes` info header of this node.
fn node_json(node: &Node) -> Value {
    json!({
        "name": node.name,
        "transport_address": "127.0.0.1:9300",
        "host": "127.0.0.1",
        "ip": "127.0.0.1",
        "version": ES_VERSION,
        "transport_version": elasticsearch_engine::version::TRANSPORT_VERSION,
        "index_version": elasticsearch_engine::version::IndexVersion::CURRENT.id(),
        "component_versions": crate::observability::component_versions(),
        "build_flavor": "default",
        "build_type": "tar",
        "build_hash": BUILD_HASH,
        "roles": ["data", "data_cold", "data_content", "data_frozen", "data_hot", "data_warm", "ingest", "master", "ml", "remote_cluster_client", "transform"],
        "attributes": crate::observability::node_attributes(),
    })
}

/// `_cluster/state` `nodes` entry (`DiscoveryNode.toXContent`).
fn state_node_json(node: &Node) -> Value {
    json!({
        "name": node.name,
        "ephemeral_id": crate::observability::EPHEMERAL_ID,
        "transport_address": crate::observability::TRANSPORT_ADDRESS,
        "external_id": node.name,
        "attributes": crate::observability::node_attributes(),
        "roles": crate::observability::ROLES,
        "version": ES_VERSION,
        "min_index_version": elasticsearch_engine::version::IndexVersion::MINIMUM_COMPATIBLE.id(),
        "max_index_version": elasticsearch_engine::version::IndexVersion::CURRENT.id()
    })
}

/// Static `_nodes` info data: ES 8.19's aggregation registry, modules and
/// ingest processor types.
static NODES_INFO: std::sync::LazyLock<Value> = std::sync::LazyLock::new(|| serde_json::from_str(include_str!("data/nodes_info.json")).unwrap_or(Value::Null));

/// The node's explicit settings (`_nodes` info `settings`).
fn node_settings(node: &Node) -> Value {
    let mut attr = Map::new();
    if let Some(o) = crate::observability::node_attributes().as_object() {
        for (k, v) in o {
            attr.insert(k.clone(), v.clone());
        }
    }
    let attr: elasticsearch_engine::settings::FlatSettings = attr.into_iter().map(|(k, v)| (format!("node.attr.{k}"), v)).collect();
    let mut flat = attr;
    let http_port = crate::http_addr(node).rsplit(':').next().unwrap_or("9200").to_string();
    let data = node.data.display().to_string();
    for (k, v) in [
        ("cluster.name", json!(node.cluster_name)),
        ("cluster.election.strategy", json!("supports_voting_only")),
        ("node.name", json!(node.name)),
        ("path.data", json!(data)),
        ("path.logs", json!(node.data.join("logs").display().to_string())),
        ("path.home", json!(std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default())),
        ("path.repo", json!([data])),
        ("discovery.type", json!("single-node")),
        ("http.type.default", json!("netty4")),
        ("http.port", json!(http_port)),
        ("transport.type.default", json!("netty4")),
        ("transport.port", json!("9300")),
        ("xpack.security.enabled", json!(if node.security.enabled() { "true" } else { "false" })),
        ("network.host", json!("127.0.0.1")),
    ] {
        flat.insert(k.to_string(), v);
    }
    // Explicit node settings, minus internal ones and those the settings
    // filter hides (passwords / secure settings).
    let node_settings = node.node_settings.read();
    let filter = elasticsearch_engine::settings_filter::SettingsFilter::for_settings(node_settings.iter());
    for (k, v) in node_settings.iter() {
        if filter.hides(k) {
            continue;
        }
        flat.insert(k.clone(), v.clone());
    }
    drop(node_settings);
    elasticsearch_engine::settings::unflatten(&flat)
}

/// Index-level blocks from `index.blocks.*` settings, keyed by ES block id.
fn index_blocks(settings: &Value) -> Map<String, Value> {
    let mut out = Map::new();
    let b = &settings["index"]["blocks"];
    let on = |k: &str| matches!(&b[k], Value::Bool(true)) || b[k].as_str() == Some("true");
    for (key, id, desc, levels) in [
        ("read_only", "5", "index read-only (api)", &["write", "metadata_write"][..]),
        ("read", "7", "index read (api)", &["read"]),
        ("write", "8", "index write (api)", &["write"]),
        ("metadata", "9", "index metadata (api)", &["metadata_read", "metadata_write"]),
        ("read_only_allow_delete", "12", "disk usage exceeded flood-stage watermark, index has read-only-allow-delete block; for more information, see https://www.elastic.co/guide/en/elasticsearch/reference/8.19/fix-watermark-errors.html", &["write", "metadata_write"]),
    ] {
        if on(key) {
            out.insert(id.to_string(), json!({"description": desc, "retryable": key == "read_only_allow_delete", "levels": levels}));
        }
    }
    out
}

/// Transport version this node reports (`nodes_versions`, index metadata).
const TRANSPORT_VERSION: &str = "8841097";

pub(crate) fn state(node: &Node, req: &Req) -> R {
    let metric_spec = req.segs.get(2).cloned().unwrap_or_else(|| "_all".into());
    // Unknown metrics are ignored (ES `ClusterState.Metric.parseString(.., true)`).
    let metrics: Vec<&str> = metric_spec.split(',').collect();
    let want = |m: &str| metrics.contains(&"_all") || metrics.contains(&m);
    // Optional index filter (`/_cluster/state/{metric}/{index}`): missing
    // indices are skipped (ES resolves leniently).
    let only: Option<std::collections::HashSet<String>> = match req.segs.get(3) {
        Some(ix) if ix != "_all" && ix != "*" => {
            let mut o = ResolveOpts::from_params(&req.params);
            if req.param("ignore_unavailable").is_none() {
                o.ignore_unavailable = true;
            }
            if req.param("allow_no_indices").is_none() {
                o.allow_no_indices = true;
            }
            Some(node.resolve(ix, o)?.into_iter().map(|t| t.svc.name.clone()).collect())
        }
        _ => None,
    };
    let filtered = only.is_some();
    let mut indices = Map::new();
    let mut routing = Map::new();
    let mut blocks = Map::new();
    let mut node_shards = Vec::new();
    let mut unassigned = Vec::new();
    for svc in node.all_indices() {
        if let Some(o) = &only {
            if !o.contains(&svc.name) {
                continue;
            }
        }
        let meta = svc.meta();
        let settings = elasticsearch_engine::settings::unflatten(&meta.settings);
        let b = index_blocks(&settings);
        if !b.is_empty() {
            blocks.insert(svc.name.clone(), Value::Object(b));
        }
        let nshards = svc.number_of_shards().max(1);
        let replicas = svc.setting_usize("index.number_of_replicas", 1);
        let mapping = meta.mappings.to_json();
        let mappings = if mapping.as_object().map(|m| m.is_empty()).unwrap_or(true) { json!({}) } else { json!({"_doc": mapping}) };
        let alloc = |i: u32| svc.shard(i).map(|c| c.allocation_id.clone()).unwrap_or_else(|| svc.uuid.clone());
        let primary_terms: Map<String, Value> = (0..nshards).map(|i| (i.to_string(), json!(1))).collect();
        let in_sync: Map<String, Value> = (0..nshards).map(|i| (i.to_string(), json!([alloc(i as u32)]))).collect();
        let hidden = meta.settings.get("index.hidden").and_then(Value::as_str) == Some("true");
        let _ = hidden;
        let mut index_md = json!({
                "version": 1, "transport_version": TRANSPORT_VERSION, "mapping_version": 1, "settings_version": 1, "aliases_version": 1,
                "routing_num_shards": nshards,
                "state": if meta.open {"open"} else {"close"},
                "settings": settings,
                "mappings": mappings,
                "aliases": meta.aliases.keys().collect::<Vec<_>>(),
                "primary_terms": primary_terms,
                "in_sync_allocations": in_sync,
                "rollover_info": {},
                "mappings_updated_version": svc.created_version().id(),
                "system": svc.name.starts_with('.'),
                "timestamp_range": {"unknown": true},
                "event_ingested_range": {"unknown": true}
            });
        // Index custom metadata (a CCR follower's `ccr` map) follows the mappings.
        if let Some(custom) = elasticsearch_engine::ccr::follower_custom(node, &svc) {
            if let Some(o) = index_md.as_object_mut() {
                let mut out = Map::new();
                for (k, v) in std::mem::take(o) {
                    let after_mappings = k == "mappings";
                    out.insert(k, v);
                    if after_mappings {
                        out.insert("ccr".into(), custom.clone());
                    }
                }
                *o = out;
            }
        }
        indices.insert(svc.name.clone(), index_md);
        let at = elasticsearch_engine::date::from_millis(svc.creation_date).format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
        let mut shards = Map::new();
        for i in 0..nshards {
            let primary = json!({"state": "STARTED", "primary": true, "node": NODE_ID, "relocating_node": null, "shard": i, "index": svc.name,
                                 "allocation_id": {"id": alloc(i as u32)}, "relocation_failure_info": {"failed_attempts": 0}});
            let mut copies = vec![primary.clone()];
            for _ in 0..replicas {
                let r = json!({"state": "UNASSIGNED", "primary": false, "node": null, "relocating_node": null, "shard": i, "index": svc.name,
                               "recovery_source": {"type": "PEER"},
                               "unassigned_info": {"reason": "INDEX_CREATED", "at": at, "delayed": false, "allocation_status": "no_attempt"},
                               "relocation_failure_info": {"failed_attempts": 0}});
                unassigned.push(r.clone());
                copies.push(r);
            }
            node_shards.push(primary);
            shards.insert(i.to_string(), Value::Array(copies));
        }
        routing.insert(svc.name.clone(), json!({ "shards": shards }));
    }
    let t = node.templates.read();
    let mut out = Map::new();
    out.insert("cluster_name".into(), json!(node.cluster_name));
    out.insert("cluster_uuid".into(), json!(node.cluster_uuid));
    if want("version") {
        out.insert("version".into(), json!(1));
        out.insert("state_uuid".into(), json!(node.cluster_uuid));
    }
    if want("master_node") {
        out.insert("master_node".into(), json!(NODE_ID));
    }
    if want("blocks") {
        out.insert("blocks".into(), if blocks.is_empty() { json!({}) } else { json!({"indices": blocks}) });
    }
    if want("nodes") {
        out.insert("nodes".into(), json!({ NODE_ID: state_node_json(node) }));
        out.insert("nodes_versions".into(), json!([{"node_id": NODE_ID, "transport_version": TRANSPORT_VERSION, "mappings_versions": {}}]));
        out.insert("nodes_features".into(), crate::features::nodes_features(NODE_ID, Some(node)));
    }
    if want("metadata") {
        let mut md = Map::new();
        md.insert("cluster_uuid".into(), json!(node.cluster_uuid));
        md.insert("cluster_uuid_committed".into(), json!(true));
        md.insert(
            "cluster_coordination".into(),
            json!({"term": 1, "last_committed_config": [NODE_ID], "last_accepted_config": [NODE_ID], "voting_config_exclusions": crate::coordination::exclusions(node)}),
        );
        md.insert("templates".into(), if filtered { json!({}) } else { json!(t.legacy) });
        md.insert("indices".into(), Value::Object(indices));
        md.insert("index-graveyard".into(), json!({"tombstones": []}));
        if filtered {
            md.insert("reserved_state".into(), json!({}));
        } else {
            if !t.index.is_empty() {
                md.insert("index_template".into(), json!({"index_template": t.index}));
            }
            if !t.component.is_empty() {
                md.insert("component_template".into(), json!({"component_template": t.component}));
            }
            let pipelines: Vec<Value> = node.ingest().get(None).into_iter().map(|(id, config)| json!({"id": id, "config": config})).collect();
            if !pipelines.is_empty() {
                md.insert("ingest".into(), json!({"pipeline": pipelines}));
            }
            md.insert("persistent_tasks".into(), json!({"last_allocation_id": 0, "tasks": []}));
            md.insert("model_registry".into(), json!({"upgraded": true, "models": {}}));
            let repos = node.snapshots.get_repositories(None).unwrap_or_else(|_| json!({}));
            if repos.as_object().map(|m| !m.is_empty()).unwrap_or(false) {
                let mut rm = Map::new();
                if let Some(m) = repos.as_object() {
                    for (name, r) in m {
                        rm.insert(name.clone(), json!({"type": r["type"], "uuid": r.get("uuid").cloned().unwrap_or(json!("_na_")), "settings": r.get("settings").cloned().unwrap_or(json!({})), "generation": 0, "pending_generation": 0}));
                    }
                }
                md.insert("repositories".into(), Value::Object(rm));
            }
            md.insert("registered_snapshots".into(), json!({"snapshots": []}));
            md.insert("reserved_state".into(), json!({"file_settings": {"version": -1, "handlers": {}, "errors": null}}));
        }
        out.insert("metadata".into(), Value::Object(md));
    }
    if want("routing_table") {
        out.insert("routing_table".into(), json!({"indices": routing}));
    }
    if want("routing_nodes") {
        out.insert("routing_nodes".into(), json!({"unassigned": unassigned, "nodes": { NODE_ID: node_shards }}));
    }
    if want("customs") {
        out.insert(
            "health".into(),
            json!({"disk": {"high_watermark": "90%", "high_max_headroom": "150gb", "flood_stage_watermark": "95%", "flood_stage_max_headroom": "100gb",
                            "frozen_flood_stage_watermark": "95%", "frozen_flood_stage_max_headroom": "20gb"},
                   "shard_limits": {"max_shards_per_node": 1000, "max_shards_per_node_frozen": 3000}}),
        );
        out.insert("snapshots".into(), json!({"snapshots": [], "node_ids_for_removal": []}));
    }
    Ok(Resp::ok(Value::Object(out)))
}

/// Machine-specific registry defaults replaced by this node's own values.
fn local_default(key: &str) -> bool {
    key.starts_with("node.attr.")
        || matches!(
            key,
            "cluster.name" | "node.name" | "node.external_id" | "node.processors" | "http.port" | "transport.port" | "discovery.type" | "path.data" | "path.home" | "path.logs"
                | "path.repo" | "network.host" | "network.bind_host" | "network.publish_host"
        )
}

fn get_cluster_settings(node: &Node, req: &Req) -> R {
    let cs = node.cluster_settings.read();
    let flat = req.flag("flat_settings")?;
    let render = |m: &Map<String, Value>| -> Value {
        let f: elasticsearch_engine::settings::FlatSettings = m.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        if flat { elasticsearch_engine::settings::as_flat_json(&f) } else { elasticsearch_engine::settings::unflatten(&f) }
    };
    let mut out = json!({"persistent": render(&cs.0), "transient": render(&cs.1)});
    if req.flag("include_defaults")? {
        // ES: node settings plus every registered default, minus what the
        // cluster settings set explicitly.
        let mut d = elasticsearch_engine::settings::cluster_defaults();
        d.retain(|k, _| !local_default(k));
        let mut ns = Map::new();
        flatten_json("", &node_settings(node), &mut ns);
        for (k, v) in ns {
            d.insert(k, v);
        }
        // Thread pool sizes follow this machine's processors.
        for p in crate::observability::pools() {
            let base = format!("thread_pool.{}", p.name);
            let vals = [("size", p.max.to_string()), ("queue_size", p.queue_size.to_string()), ("core", p.core.to_string()), ("max", p.max.to_string())];
            for (attr, v) in vals {
                let k = format!("{base}.{attr}");
                if d.contains_key(&k) {
                    d.insert(k, json!(v));
                }
            }
        }
        d.insert("node.processors".into(), json!(format!("{:.1}", crate::sysinfo::sys().cpus as f64)));
        d.insert("node.external_id".into(), json!(node.name));
        for k in ["network.bind_host", "network.publish_host"] {
            d.insert(k.into(), json!(["127.0.0.1"]));
        }
        {
            // ES's `SettingsFilter`: no secure (keystore) or filtered setting.
            let node_settings = node.node_settings.read();
            let filter = elasticsearch_engine::settings_filter::SettingsFilter::for_settings(node_settings.iter());
            for (k, v) in node_settings.iter().filter(|(k, _)| !filter.hides(k)) {
                d.insert(k.clone(), json!(v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())));
            }
        }
        d.retain(|k, _| !cs.0.contains_key(k) && !cs.1.contains_key(k));
        let dm: Map<String, Value> = d.into_iter().collect();
        out["defaults"] = render(&dm);
    }
    Ok(Resp::ok(out))
}

fn put_cluster_settings(node: &Node, req: &Req) -> R {
    let body = req.json()?;
    let empty = |k: &str| body.get(k).and_then(Value::as_object).map(|m| m.is_empty()).unwrap_or(true);
    if !body.is_object() || (empty("persistent") && empty("transient")) {
        return Err(EsError::action_validation(&["no settings to update"]));
    }
    let o = body.as_object().ok_or_else(|| EsError::action_validation(&["no settings to update"]))?;
    // Validate everything before applying anything (transient first, like ES).
    for kind in ["transient", "persistent"] {
        if let Some(v) = o.get(kind) {
            let flat = elasticsearch_engine::settings::flatten(v);
            elasticsearch_engine::settings::validate_cluster_update(kind, &flat)?;
            if let Some(Value::String(r)) = flat.get("repositories.default_repository") {
                crate::snapshot_api::validate_default_repository(node, r)?;
            }
        }
    }
    let mut out = json!({"acknowledged": true, "persistent": {}, "transient": {}});
    {
        let mut committed = node.cluster_settings.write();
        let mut cs = committed.clone();
        for (kind, idx) in [("persistent", 0), ("transient", 1)] {
            if let Some(v) = o.get(kind) {
                let flat = elasticsearch_engine::settings::flatten(v);
                let target = if idx == 0 { &mut cs.0 } else { &mut cs.1 };
                let mut applied = Map::new();
                for (k, val) in flat {
                    if val.is_null() {
                        // `a.*: null` resets every setting under the prefix.
                        match k.strip_suffix('*') {
                            Some(prefix) => target.retain(|key, _| !key.starts_with(prefix)),
                            None => {
                                target.remove(&k);
                            }
                        }
                    } else {
                        target.insert(k.clone(), val.clone());
                        applied.insert(k, val);
                    }
                }
                let af: elasticsearch_engine::settings::FlatSettings = applied.into_iter().collect();
                out[kind] = if req.flag("flat_settings")? { elasticsearch_engine::settings::as_flat_json(&af) } else { elasticsearch_engine::settings::unflatten(&af) };
            }
        }
        let mut effective: std::collections::BTreeMap<String, Value> = node.node_settings.read().iter().map(|(k,v)|(k.clone(),v.clone())).collect();
        effective.extend(cs.0.iter().map(|(k,v)|(k.clone(),v.clone())));
        effective.extend(cs.1.iter().map(|(k,v)|(k.clone(),v.clone())));
        elasticsearch_engine::script::rate_limit::validate_settings(&effective)?;
        *committed = cs;
    }
    node.persist_state()?;
    // Applied once committed.
    elasticsearch_engine::memory::configure(node);
    elasticsearch_engine::script::rate_limit::configure(node)?;
    Ok(Resp::ok(out))
}

const NODE_INFO_METRICS: &[&str] = &[
    "settings", "os", "process", "jvm", "thread_pool", "transport", "http", "remote_cluster_server", "plugins", "ingest", "aggregations", "indices",
];
fn node_matches(node: &Node, spec: &str) -> bool {
    spec.split(',').map(str::trim).any(|s| match s {
        "" | "_all" | "_local" | "_master" | "*" => true,
        _ => {
            if let Some((_, v)) = s.split_once(':') {
                v != "false"
            } else {
                s == NODE_ID || s == node.name || elasticsearch_engine::mapping::wildcard_match(s, &node.name) || elasticsearch_engine::mapping::wildcard_match(s, NODE_ID)
            }
        }
    })
}

fn nodes_envelope(node: &Node, nodes: Map<String, Value>) -> Value {
    let n = nodes.len();
    json!({"_nodes": {"total": n, "successful": n, "failed": 0}, "cluster_name": node.cluster_name, "nodes": nodes})
}

fn flatten_json(prefix: &str, v: &Value, out: &mut Map<String, Value>) {
    match v {
        Value::Object(m) => m.iter().for_each(|(k, x)| flatten_json(&if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") }, x, out)),
        other => {
            out.insert(prefix.to_string(), other.clone());
        }
    }
}

fn nodes(node: &Node, req: &Req) -> R {
    // /_nodes, /_nodes/{node_id|metric}, /_nodes/{node_id}/{metric}, /_nodes/{node_id}/info/{metric}
    let rest: Vec<&str> = req.segs.iter().skip(1).map(String::as_str).collect();
    let is_metric_list = |s: &str| s.split(',').all(|m| m == "_all" || m == "_none" || NODE_INFO_METRICS.contains(&m));
    let (node_spec, metric_spec) = match rest.as_slice() {
        [] => ("_all", "_all"),
        [x] if is_metric_list(x) => ("_all", *x),
        [x] => (*x, "_all"),
        [n, "info", m, ..] => (*n, *m),
        [n, m, ..] => (*n, *m),
    };
    // ES `RestNodesInfoAction` disregards unknown metrics ("disregard
    // unknown metrics; TODO eschew this lenience?": `retainAll`): ECK asks
    // for `/_nodes/_all/no-metrics` to get the bare node entries.
    let selected = node_matches(node, node_spec);
    let metrics: Vec<&str> = metric_spec.split(',').filter(|m| *m == "_all" || *m == "_none" || NODE_INFO_METRICS.contains(m)).collect();
    let want = |m: &str| metrics.contains(&"_all") || metrics.contains(&m);
    let s = crate::sysinfo::sys();
    let (_, heap_max) = crate::sysinfo::heap(&s);
    let cpus = s.cpus;
    let mut n = node_json(node);
    if want("indices") {
        if let Some(o) = n.as_object_mut() {
            // ES prints the indexing buffer right after the build fields.
            let mut out = Map::new();
            for (k, v) in std::mem::take(o) {
                let after = k == "build_hash";
                out.insert(k, v);
                if after {
                    out.insert("total_indexing_buffer".into(), json!(elasticsearch_engine::memory::indexing_buffer()));
                }
            }
            *o = out;
        }
    }
    if want("settings") {
        let st = node_settings(node);
        n["settings"] = if req.flag("flat_settings")? {
            let mut flat = Map::new();
            flatten_json("", &st, &mut flat);
            Value::Object(flat)
        } else {
            st
        };
    }
    if want("os") {
        let arch = match std::env::consts::ARCH {
            "x86_64" => "amd64",
            a => a,
        };
        n["os"] = json!({"refresh_interval_in_millis": 1000, "name": s.os_name, "pretty_name": s.os_pretty_name, "arch": arch, "version": s.os_version, "available_processors": cpus, "allocated_processors": cpus});
    }
    if want("process") {
        n["process"] = json!({"refresh_interval_in_millis": 1000, "id": std::process::id(), "mlockall": crate::sysinfo::memory_locked()});
    }
    if want("jvm") {
        let start = elasticsearch_engine::date::now_millis() - node.uptime_millis() as i64;
        n["jvm"] = json!({"pid": std::process::id(), "version": "21.0.2", "vm_name": "OpenJDK 64-Bit Server VM", "vm_version": "21.0.2+13-58", "vm_vendor": "Oracle Corporation", "using_bundled_jdk": true, "start_time_in_millis": start,
                          "mem": {"heap_init_in_bytes": heap_max, "heap_max_in_bytes": heap_max, "non_heap_init_in_bytes": 7667712, "non_heap_max_in_bytes": 0, "direct_max_in_bytes": heap_max / 2},
                          "gc_collectors": [],
                          "runtime": "native", "jvm_measurements_available": false,
                          "memory_pools": [],
                          "using_compressed_ordinary_object_pointers": "true", "input_arguments": []});
    }
    if want("thread_pool") {
        n["thread_pool"] = crate::observability::thread_pool_info();
    }
    if want("transport") {
        n["transport"] = json!({"bound_address": ["127.0.0.1:9300"], "publish_address": "127.0.0.1:9300", "profiles": {}});
    }
    if want("http") {
        let addr = crate::http_addr(node);
        n["http"] = json!({"bound_address": [addr], "publish_address": addr, "max_content_length_in_bytes": 104857600});
    }
    if want("plugins") {
        n["plugins"] = json!([]);
        n["modules"] = NODES_INFO["modules"].clone();
    }
    if want("ingest") {
        let procs: Vec<Value> = NODES_INFO["ingest_processors"].as_array().map(|a| a.iter().map(|t| json!({"type": t})).collect()).unwrap_or_default();
        n["ingest"] = json!({"processors": procs});
    }
    if want("aggregations") {
        n["aggregations"] = NODES_INFO["aggregations"].clone();
    }
    let mut out = Map::new();
    if selected {
        out.insert(NODE_ID.to_string(), n);
    }
    Ok(Resp::ok(nodes_envelope(node, out)))
}

/// `GET /_info/{target}`.
fn cluster_info(node: &Node, req: &Req) -> R {
    const TARGETS: &[&str] = &["http", "ingest", "thread_pool", "script"];
    let spec = req.seg(1);
    let targets: Vec<&str> = spec.split(',').filter(|s| !s.is_empty()).collect();
    if targets.is_empty() {
        return crate::not_found_handler(req);
    }
    if targets.contains(&"_all") && targets.len() > 1 {
        return Err(EsError::illegal_argument(format!("request [{}] contains _all and individual target [{spec}]", req.path)));
    }
    let bad: Vec<&str> = targets.iter().copied().filter(|t| *t != "_all" && !TARGETS.contains(t)).collect();
    if !bad.is_empty() {
        return Err(EsError::illegal_argument(format!(
            "request [{}] contains unrecognized target{}: [{}]",
            req.path,
            if bad.len() > 1 { "s" } else { "" },
            bad.join(", ")
        )));
    }
    let want = |t: &str| targets.contains(&"_all") || targets.contains(&t);
    let mut out = Map::new();
    out.insert("cluster_name".into(), json!(node.cluster_name));
    if want("http") {
        out.insert("http".into(), crate::observability::http_section());
    }
    if want("ingest") {
        out.insert("ingest".into(), node.ingest().stats_json());
    }
    if want("thread_pool") {
        out.insert("thread_pool".into(), crate::observability::thread_pool_section(node));
    }
    if want("script") {
        out.insert("script".into(), json!({"compilations": 0, "cache_evictions": 0, "compilation_limit_triggered": 0, "compilations_history": {"5m": 0, "15m": 0, "24h": 0}, "contexts": []}));
    }
    Ok(Resp::ok(Value::Object(out)))
}
