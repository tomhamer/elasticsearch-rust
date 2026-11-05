//! `GET /_capabilities` (ES 8.19 `RestNodesCapabilitiesAction`).
//!
//! Answers whether this cluster's REST layer has a handler for
//! `method` + `path` that accepts every listed `parameters` entry and
//! implements every listed `capabilities` entry. Elastic's YAML REST tests
//! gate on it (`requires: capabilities`).
//!
//! The table below is deliberately conservative: an entry is listed only when
//! the behaviour the capability names was checked in the code that serves the
//! endpoint. Anything not listed answers `supported: false`, which is also
//! what ES answers for an unknown path, parameter or capability. Every node
//! runs the same binary, so the answer is the same on all of them and
//! `supported` is never `null`.

use crate::front::{RestReq, RestResp};
use crate::node::ClusterNode;
use serde_json::json;
use std::sync::Arc;

struct Entry {
    methods: &'static [&'static str],
    /// `{}` matches any single path segment (the request may name it
    /// `{index}`, `{target}`, ... or give a concrete value).
    path: &'static str,
    params: &'static [&'static str],
    caps: &'static [&'static str],
}

/// Parameters every REST handler accepts.
const GENERIC_PARAMS: &[&str] = &["pretty", "human", "error_trace", "filter_path", "format", "source", "source_content_type"];

const SEARCH_CAPS: &[&str] = &[
    "exclude_source_vectors_setting",
    "pipeline_aggs_parent_multi_bucket_error",
    // `rescore_vector.oversample` on quantized dense_vector index options and on knn search.
    "knn_quantized_vector_rescore",
    "knn_quantized_vector_rescore_oversample",
    // highlight `number_of_fragments` is rejected outside [0, 10000].
    "highlight_number_of_fragments_bounded",
    // highlight `max_analyzed_offset: -1` falls back to `index.highlight.max_analyzed_offset`.
    "highlight_max_analyzed_offset_default",
];

const LIFECYCLE_CAPS: &[&str] = &["data_stream_global_retention", "data_stream_lifecycle_effective_retention"];

const TABLE: &[Entry] = &[
    // Coordinator and primary failure-store paths preserve used/not_enabled/
    // failed status, and acknowledge the actual destination shard copies.
    Entry { methods: &["POST"], path: "/{}/_doc", params: &[], caps: &["failure_store_status"] },
    Entry { methods: &["POST", "PUT"], path: "/_bulk", params: &[], caps: &["failure_store_status"] },
    Entry { methods: &["POST", "PUT"], path: "/{}/_bulk", params: &[], caps: &["failure_store_status"] },
    // Snapshot service filters completed and running snapshots before pagination.
    Entry { methods: &["GET"], path: "/_snapshot/{}/{}", params: &["state"], caps: &[] },
    // The coordinator's snapshot service deletes synchronously or not (`wait_for_completion`).
    Entry { methods: &["DELETE"], path: "/_snapshot/{}/{}", params: &["wait_for_completion", "master_timeout"], caps: &[] },
    Entry { methods: &["GET", "POST"], path: "/_cluster/allocation/explain", params: &["index", "shard", "primary", "current_node", "include_yes_decisions", "include_disk_info", "master_timeout"], caps: &["query_parameter_support"] },
    // Coordinator registration validation and default-repository mutation guards.
    Entry { methods: &["PUT"], path: "/_cluster/settings", params: &[], caps: &["repositories.default_repository_setting"] },
    Entry { methods: &["GET"], path: "/_capabilities", params: &["method", "path", "parameters", "capabilities", "local_only", "timeout"], caps: &[] },
    // Health reports `unassigned_primary_shards`.
    Entry { methods: &["GET"], path: "/_cluster/health", params: &[], caps: &["unassigned_pri_shard_count"] },
    Entry { methods: &["GET"], path: "/_cluster/health/{}", params: &[], caps: &["unassigned_pri_shard_count"] },
    // These answer 200 with an empty, non-JSON body.
    Entry { methods: &["POST", "DELETE"], path: "/_cluster/voting_config_exclusions", params: &[], caps: &["plain_text_empty_response"] },
    Entry { methods: &["DELETE"], path: "/_internal/desired_balance", params: &[], caps: &["plain_text_empty_response"] },
    // The coordinator's desired balance reports per-node weights.
    Entry { methods: &["GET"], path: "/_internal/desired_balance", params: &["master_timeout"], caps: &["desired_balance_node_weights_in_response"] },
    // The coordinator's cat nodes has the `available_processors` (`ap`) column.
    Entry { methods: &["GET"], path: "/_cat/nodes", params: &[], caps: &["available_processors_availability"] },
    // The coordinator's cat segments lists closed indices with `allow_closed`.
    Entry { methods: &["GET"], path: "/_cat/segments", params: &[], caps: &["allow_closed"] },
    Entry { methods: &["GET"], path: "/_cat/segments/{}", params: &[], caps: &["allow_closed"] },
    Entry { methods: &["DELETE"], path: "/_internal/desired_nodes", params: &[], caps: &["plain_text_empty_response"] },
    // The coordinator renders cluster stats (human totals, dense_vector
    // mapping stats, CCS stats) as the engine does.
    Entry { methods: &["GET"], path: "/_cluster/stats", params: &["flat_settings", "timeout", "include_remotes"], caps: &["human-readable-total-docs-size", "verbose-dense-vector-mapping-stats", "ccs-stats"] },
    // Each node's entry is its own engine's rendering (dense_vector off-heap stats included).
    Entry { methods: &["GET"], path: "/_nodes/stats", params: &["completion_fields", "fielddata_fields", "fields", "groups", "level", "types", "timeout", "include_segment_file_sizes", "include_unloaded_segments"], caps: &["dense_vector_off_heap_stats"] },
    Entry { methods: &["GET", "POST"], path: "/_search", params: &[], caps: SEARCH_CAPS },
    Entry { methods: &["GET", "POST"], path: "/{}/_search", params: &[], caps: SEARCH_CAPS },
    Entry { methods: &["GET"], path: "/_data_stream/{}/_lifecycle", params: &[], caps: LIFECYCLE_CAPS },
];

fn segs(path: &str) -> Vec<&str> {
    path.split('/').filter(|s| !s.is_empty()).collect()
}

fn path_matches(pattern: &str, path: &str) -> bool {
    let (p, q) = (segs(pattern), segs(path));
    p.len() == q.len()
        && p.iter().zip(q.iter()).all(|(a, b)| {
            if *a == "{}" {
                // A wildcard never swallows a literal API segment such as `_search`.
                b.starts_with('{') || !b.starts_with('_')
            } else {
                a == b
            }
        })
}

fn list(v: Option<&String>) -> Vec<String> {
    v.map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default()
}

/// The truth table, separated from the HTTP plumbing for tests.
pub fn supported(method: &str, path: &str, params: &[String], caps: &[String]) -> bool {
    // ES|QL routes answer from the engine's table (release 9.5.5 capability list).
    if path == "/_query" || path.starts_with("/_query/") {
        let (p, c): (Vec<&str>, Vec<&str>) = (params.iter().map(String::as_str).collect(), caps.iter().map(String::as_str).collect());
        return elasticsearch_http::capability_supported(method, path, &p, &c) == Some(serde_json::Value::Bool(true));
    }
    // Metadata routes use the engine handler on the local node or elected
    // master, so answer from that handler's table rather than hiding its caps.
    if let Ok(m) = axum::http::Method::from_bytes(method.as_bytes()) {
        let segments = segs(path);
        if matches!(crate::front::classify::classify(&m, &segments), crate::front::classify::Route::Local | crate::front::classify::Route::Master) {
            let (p, c): (Vec<&str>, Vec<&str>) = (params.iter().map(String::as_str).collect(), caps.iter().map(String::as_str).collect());
            return elasticsearch_http::capability_supported(method, path, &p, &c) == Some(serde_json::Value::Bool(true));
        }
    }
    let route = |e: &&Entry| e.methods.iter().any(|m| m.eq_ignore_ascii_case(method)) && path_matches(e.path, path);
    let cluster = TABLE.iter().filter(route).any(|e| {
        params.iter().all(|p| e.params.contains(&p.as_str()) || GENERIC_PARAMS.contains(&p.as_str()))
            && caps.iter().all(|c| e.caps.contains(&c.as_str()))
    });
    // A distributed route the coordinator serves runs the same engine
    // handler on every shard copy, so it implements what that handler's
    // table lists too; the cluster table adds what only the coordinator
    // provides.
    cluster || TABLE.iter().any(|e| route(&e)) && {
        let (p, c): (Vec<&str>, Vec<&str>) = (params.iter().map(String::as_str).collect(), caps.iter().map(String::as_str).collect());
        elasticsearch_http::capability_supported(method, path, &p, &c) == Some(serde_json::Value::Bool(true))
    }
}

pub fn handle(node: &Arc<ClusterNode>, req: &RestReq) -> RestResp {
    if req.method != "GET" {
        return RestResp::error(405, "method_not_allowed_exception", &format!("Incorrect HTTP method for uri [{}] and method [{}], allowed: [GET]", req.uri, req.method));
    }
    let q = req.params();
    let method = q.get("method").map(|s| s.to_ascii_uppercase()).unwrap_or_else(|| "GET".to_string());
    let path = q.get("path").cloned().unwrap_or_else(|| "/".to_string());
    let ok = supported(&method, &path, &list(q.get("parameters")), &list(q.get("capabilities")));
    let st = node.state();
    let total = st.nodes.len();
    RestResp::json(200, &json!({
        "_nodes": {"total": total, "successful": total, "failed": 0},
        "cluster_name": st.cluster_name,
        "supported": ok,
    }))
}

#[cfg(test)]
mod tests {
    use super::supported;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn table() {
        assert!(supported("GET", "/_snapshot/{repository}/{snapshot}", &s(&["state"]), &[]));
        assert!(supported("PUT", "/_cluster/settings", &[], &s(&["repositories.default_repository_setting"])));
        assert!(supported("GET", "/_capabilities", &s(&["method", "path", "parameters", "capabilities"]), &[]));
        assert!(!supported("GET", "/_capabilities", &s(&["unknown"]), &[]));
        assert!(supported("GET", "/_cluster/health", &[], &s(&["unassigned_pri_shard_count"])));
        assert!(supported("GET", "/_cluster/stats", &[], &s(&["human-readable-total-docs-size", "ccs-stats"])));
        assert!(supported("GET", "/_nodes/stats", &[], &s(&["dense_vector_off_heap_stats"])));
        assert!(supported("DELETE", "/_snapshot/{repository}/{snapshot}", &s(&["wait_for_completion"]), &[]));
        assert!(supported("GET", "/_data_stream/{index}/_lifecycle", &[], &s(&["data_stream_global_retention"])));
        assert!(supported("GET", "/_data_stream/{target}/_lifecycle", &[], &s(&["data_stream_global_retention"])));
        assert!(supported("GET", "/_data_stream/{target}", &[], &s(&["failure_store.lifecycle"])));
        assert!(!supported("GET", "/_data_stream/{target}", &[], &s(&["unknown_capability"])));
        assert!(supported("DELETE", "/_internal/desired_nodes", &[], &s(&["plain_text_empty_response"])));
        assert!(supported("GET", "/_internal/desired_balance", &[], &s(&["desired_balance_node_weights_in_response"])));
        assert!(supported("GET", "/_cat/segments", &[], &s(&["allow_closed"])));
        assert!(supported("GET", "/_cat/nodes", &[], &s(&["available_processors_availability"])));
        assert!(supported("GET", "/_cat/segments", &s(&["bytes", "allow_closed"]), &[]));
        assert!(supported("PUT", "/{index}", &[], &s(&["logsdb_index_mode"])));
        assert!(supported("GET", "/_search", &[], &s(&["exclude_vectors_param"])));
        assert!(supported("POST", "/_search", &[], &s(&["aggregate_metric_double_defaults_to_average"])));
        assert!(!supported("GET", "/_search", &[], &s(&["unknown_capability"])));
        assert!(!supported("PUT", "/_search", &[], &[]));
        assert!(supported("POST", "/_query", &s(&["format"]), &s(&["inline_stats", "fn_abs"])));
        assert!(!supported("POST", "/_query", &[], &s(&["optional_fields_v2"])));
    }
}
