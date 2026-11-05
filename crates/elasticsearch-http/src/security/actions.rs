//! Map a REST request to the ES transport action it executes and where its
//! index expression lives (ES authorizes actions, not URLs).

use crate::Req;
use axum::http::Method;

#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    /// Cluster action (or an index action without indices).
    None,
    /// Index expression at `segs[i]`; when missing, the request means `_all`
    /// and the authorized names are inserted at `i`.
    Seg(usize),
    /// `_bulk` (per item), `_mget` (per doc), `_msearch` (per search).
    Bulk,
    Mget,
    Msearch,
    /// `_reindex`: read on source, write on dest (from the body).
    Reindex,
    /// `POST /_aliases`: indices and aliases from the body.
    AliasesBody,
    /// Index names only known to the engine (ES|QL, SQL, templates):
    /// require the action on some index.
    AnyIndex,
    /// Any authenticated user (scroll / async results bound to their owner).
    Authenticated,
    /// Fixed (system) index names the action's request targets
    /// (`IndicesRequest.indices()` of the connector APIs).
    Fixed(&'static [&'static str]),
}

#[derive(Debug, Clone)]
pub struct Act {
    pub action: String,
    pub target: Target,
    /// Second action checked on the same names (document writes: `op_type`).
    pub item: Option<String>,
    /// Writes that auto-create a missing index also need `indices:admin/auto_create`.
    pub auto_create: bool,
    /// Alias names (from `/{index}/_alias/{name}`) that also need the action.
    pub alias_seg: Option<usize>,
}

fn act(a: impl Into<String>, t: Target) -> Act {
    Act { action: a.into(), target: t, item: None, auto_create: false, alias_seg: None }
}

fn cluster(a: impl Into<String>) -> Act {
    act(a, Target::None)
}

fn is_read(m: &Method) -> bool {
    *m == Method::GET || *m == Method::HEAD
}

fn crud(m: &Method, base: &str) -> Act {
    match *m {
        Method::GET | Method::HEAD => cluster(format!("{base}/get")),
        Method::DELETE => cluster(format!("{base}/delete")),
        _ => cluster(format!("{base}/put")),
    }
}

fn by_method(m: &Method, get: &str, delete: &str, put: &str) -> String {
    match *m {
        Method::GET | Method::HEAD => get,
        Method::DELETE => delete,
        _ => put,
    }
    .to_string()
}

fn by_method_str(m: &Method, get: &'static str, delete: &'static str, put: &'static str) -> &'static str {
    match *m {
        Method::GET | Method::HEAD => get,
        Method::DELETE => delete,
        _ => put,
    }
}

const CONNECTORS: &[&str] = &[".elastic-connectors-v1"];
const SYNC_JOBS: &[&str] = &[".elastic-connectors-sync-jobs-v1"];
const CONNECTORS_AND_SYNC_JOBS: &[&str] = &[".elastic-connectors-sync-jobs-v1", ".elastic-connectors-v1"];

/// The connector APIs (`_connector`, `_connector/_sync_job`,
/// `_connector/_secret`): ES's connector actions are index actions on the
/// connector system indices (`ConnectorActionRequest.indices()`).
fn connector(m: &Method, s: &[&str]) -> Act {
    let g = |i: usize| s.get(i).copied().unwrap_or("");
    match g(1) {
        "_secret" => cluster(match *m {
            Method::GET | Method::HEAD => "cluster:admin/xpack/connector/secret/get",
            Method::DELETE => "cluster:admin/xpack/connector/secret/delete",
            Method::PUT => "cluster:admin/xpack/connector/secret/put",
            _ => "cluster:admin/xpack/connector/secret/post",
        }),
        "_sync_job" => {
            let (a, t) = match (g(2), g(3)) {
                ("", _) if is_read(m) => ("indices:data/read/xpack/connector/sync_job/list", SYNC_JOBS),
                ("", _) => ("indices:data/write/xpack/connector/sync_job/post", CONNECTORS_AND_SYNC_JOBS),
                (_, "_cancel") => ("indices:data/write/xpack/connector/sync_job/cancel", SYNC_JOBS),
                (_, "_check_in") => ("indices:data/write/xpack/connector/sync_job/check_in", SYNC_JOBS),
                (_, "_claim") => ("indices:data/write/xpack/connector/sync_job/claim", SYNC_JOBS),
                (_, "_error") => ("indices:data/write/xpack/connector/sync_job/update_error", SYNC_JOBS),
                (_, "_stats") => ("indices:data/write/xpack/connector/sync_job/update_stats", SYNC_JOBS),
                _ => (by_method_str(m, "indices:data/read/xpack/connector/sync_job/get", "indices:data/write/xpack/connector/sync_job/delete", "indices:data/write/xpack/connector/sync_job/get"), SYNC_JOBS),
            };
            act(a, Target::Fixed(t))
        }
        "" if is_read(m) => act("indices:data/read/xpack/connector/list", Target::Fixed(CONNECTORS)),
        "" if *m == Method::POST => act("indices:data/write/xpack/connector/post", Target::Fixed(CONNECTORS)),
        "" => act("indices:data/write/xpack/connector/put", Target::Fixed(CONNECTORS)),
        _ => {
            let a = match (g(2), g(3)) {
                ("", _) => match *m {
                    Method::GET | Method::HEAD => "indices:data/read/xpack/connector/get",
                    Method::DELETE => return act("indices:data/write/xpack/connector/delete", Target::Fixed(CONNECTORS_AND_SYNC_JOBS)),
                    _ => "indices:data/write/xpack/connector/put",
                },
                ("_api_key_id", _) => "indices:data/write/xpack/connector/update_api_key_id",
                ("_check_in", _) => "indices:data/write/xpack/connector/update_last_seen",
                ("_configuration", _) => "indices:data/write/xpack/connector/update_configuration",
                ("_error", _) => "indices:data/write/xpack/connector/update_error",
                ("_features", _) => "indices:data/write/xpack/connector/update_features",
                ("_filtering", "_activate") => "indices:data/write/xpack/connector/update_filtering/activate",
                ("_filtering", "_validation") => "indices:data/write/xpack/connector/update_filtering/draft_validation",
                ("_filtering", _) => "indices:data/write/xpack/connector/update_filtering",
                ("_index_name", _) => "indices:data/write/xpack/connector/update_index_name",
                ("_last_sync", _) => "indices:data/write/xpack/connector/update_last_sync_stats",
                ("_name", _) => "indices:data/write/xpack/connector/update_name",
                ("_native", _) => "indices:data/write/xpack/connector/update_native",
                ("_pipeline", _) => "indices:data/write/xpack/connector/update_pipeline",
                ("_scheduling", _) => "indices:data/write/xpack/connector/update_scheduling",
                ("_service_type", _) => "indices:data/write/xpack/connector/update_service_type",
                ("_status", _) => "indices:data/write/xpack/connector/update_status",
                _ => "indices:data/write/xpack/connector/put",
            };
            act(a, Target::Fixed(CONNECTORS))
        }
    }
}

/// Classify the request (segments as dispatched).
pub fn classify(req: &Req) -> Act {
    let s: Vec<&str> = req.segs.iter().map(String::as_str).collect();
    let m = &req.method;
    let g = |i: usize| s.get(i).copied().unwrap_or("");
    if s.is_empty() {
        return cluster("cluster:monitor/main");
    }
    let first = s[0];
    if !first.starts_with('_') || first == "_all" {
        return classify_index(req, &s);
    }
    match first {
        "_search" | "_count" => match g(1) {
            "scroll" => act("indices:data/read/scroll", Target::Authenticated),
            "template" => act("indices:data/read/search/template", Target::Seg(0)),
            _ => act("indices:data/read/search", Target::Seg(0)),
        },
        "_msearch" => act(if g(1) == "template" { "indices:data/read/msearch/template" } else { "indices:data/read/msearch" }, Target::Msearch),
        "_bulk" => act("indices:data/write/bulk", Target::Bulk),
        "_mget" => act("indices:data/read/mget", Target::Mget),
        "_pit" => act("indices:data/read/close_point_in_time", Target::Authenticated),
        "_field_caps" => act("indices:data/read/field_caps", Target::Seg(0)),
        "_validate" => act("indices:admin/validate/query", Target::Seg(0)),
        "_terms_enum" => act("indices:data/read/xpack/termsenum/list", Target::Seg(0)),
        "_search_shards" => act("indices:admin/shards/search_shards", Target::Seg(0)),
        "_refresh" => act("indices:admin/refresh", Target::Seg(0)),
        "_flush" => act("indices:admin/flush", Target::Seg(0)),
        "_forcemerge" => act("indices:admin/forcemerge", Target::Seg(0)),
        "_cache" => act("indices:admin/cache/clear", Target::Seg(0)),
        "_stats" => act("indices:monitor/stats", Target::Seg(0)),
        "_segments" => act("indices:monitor/segments", Target::Seg(0)),
        "_recovery" => act("indices:monitor/recovery", Target::Seg(0)),
        "_shard_stores" => act("indices:monitor/shard_stores", Target::Seg(0)),
        "_disk_usage" => act("indices:admin/analyze_disk_usage", Target::Seg(0)),
        "_settings" => act(if is_read(m) { "indices:monitor/settings/get" } else { "indices:admin/settings/update" }, Target::Seg(0)),
        "_mapping" | "_mappings" => act(if g(1) == "field" { "indices:admin/mappings/fields/get" } else if is_read(m) { "indices:admin/mappings/get" } else { "indices:admin/mapping/put" }, Target::Seg(0)),
        "_alias" => act(if is_read(m) { "indices:admin/aliases/get" } else { "indices:admin/aliases" }, Target::Seg(0)),
        "_aliases" => {
            if is_read(m) {
                act("indices:admin/aliases/get", Target::Seg(0))
            } else {
                act("indices:admin/aliases", Target::AliasesBody)
            }
        }
        "_analyze" => act("indices:admin/analyze", Target::AnyIndex),
        "_explain" => act("indices:data/read/explain", Target::Seg(0)),
        "_rank_eval" => act("indices:data/read/rank_eval", Target::Seg(0)),
        "_async_search" => {
            if *m == Method::POST && g(1).is_empty() {
                act("indices:data/read/async_search/submit", Target::Seg(0))
            } else if g(1) == "status" {
                cluster("cluster:monitor/async_search/status")
            } else {
                act("indices:data/read/async_search/get", Target::Authenticated)
            }
        }
        "_resolve" => match g(1) {
            "cluster" => act("indices:admin/resolve/cluster", Target::Seg(2)),
            _ => act("indices:admin/resolve/index", Target::Seg(2)),
        },
        "_reindex" => {
            if g(2) == "_rethrottle" {
                cluster("cluster:admin/reindex/rethrottle")
            } else {
                act("indices:data/write/reindex", Target::Reindex)
            }
        }
        "_update_by_query" | "_delete_by_query" => cluster("cluster:admin/reindex/rethrottle"),
        "_query" => {
            if g(1) == "async" {
                act("indices:data/read/esql/async/get", Target::Authenticated)
            } else {
                act("indices:data/read/esql", Target::AnyIndex)
            }
        }
        "_sql" => act("indices:data/read/sql", Target::AnyIndex),
        "_eql" => act("indices:data/read/eql", Target::Authenticated),
        "_render" => act("indices:data/read/search/template", Target::AnyIndex),
        "_cluster" => match g(1) {
            "health" => cluster("cluster:monitor/health"),
            "state" => cluster("cluster:monitor/state"),
            "settings" => cluster(if is_read(m) { "cluster:monitor/settings" } else { "cluster:admin/settings/update" }),
            "stats" => cluster("cluster:monitor/stats"),
            "pending_tasks" => cluster("cluster:monitor/task"),
            "allocation" => cluster("cluster:monitor/allocation/explain"),
            "reroute" => cluster("cluster:admin/reroute"),
            "voting_config_exclusions" => cluster("cluster:admin/voting_config/add_exclusions"),
            "info" => cluster("cluster:monitor/info"),
            other => cluster(if is_read(m) { format!("cluster:monitor/{other}") } else { format!("cluster:admin/{other}") }),
        },
        "_nodes" => {
            let all: Vec<&str> = s.iter().skip(1).copied().collect();
            if all.contains(&"stats") {
                cluster("cluster:monitor/nodes/stats")
            } else if all.contains(&"hot_threads") || all.contains(&"hotthreads") {
                cluster("cluster:monitor/nodes/hot_threads")
            } else if all.contains(&"usage") {
                cluster("cluster:monitor/nodes/usage")
            } else if all.contains(&"reload_secure_settings") {
                cluster("cluster:admin/nodes/reload_secure_settings")
            } else {
                cluster("cluster:monitor/nodes/info")
            }
        }
        "_cat" => match g(1) {
            "count" => act("indices:data/read/search", Target::Seg(2)),
            "indices" => act("indices:monitor/settings/get", Target::Seg(2)),
            "aliases" => act("indices:admin/aliases/get", Target::AnyIndex),
            "shards" => act("indices:monitor/stats", Target::Seg(2)),
            "segments" => act("indices:monitor/segments", Target::Seg(2)),
            "recovery" => act("indices:monitor/recovery", Target::Seg(2)),
            "health" => cluster("cluster:monitor/health"),
            "tasks" => cluster("cluster:monitor/tasks/lists"),
            "pending_tasks" => cluster("cluster:monitor/task"),
            "repositories" => cluster("cluster:admin/repository/get"),
            "snapshots" => cluster("cluster:admin/snapshot/get"),
            "fielddata" => cluster("cluster:monitor/nodes/stats"),
            "nodes" | "thread_pool" | "plugins" | "nodeattrs" => cluster("cluster:monitor/nodes/info"),
            "ml" => cluster("cluster:monitor/xpack/ml/job/stats/get"),
            "transforms" => cluster("cluster:monitor/transform/stats/get"),
            "" => act("cluster:monitor/main", Target::Authenticated),
            _ => cluster("cluster:monitor/state"),
        },
        "_xpack" => cluster(if g(1) == "usage" { "cluster:monitor/xpack/usage" } else { "cluster:monitor/xpack/info" }),
        "_license" => match g(1) {
            "basic_status" => cluster("cluster:monitor/xpack/license/basic_status"),
            "trial_status" => cluster("cluster:monitor/xpack/license/trial_status"),
            "start_basic" => cluster("cluster:admin/xpack/license/start_basic"),
            "start_trial" => cluster("cluster:admin/xpack/license/start_trial"),
            "feature_usage" => cluster("cluster:monitor/xpack/license/feature_usage"),
            _ => cluster(by_method(m, "cluster:monitor/xpack/license/get", "cluster:admin/xpack/license/delete", "cluster:admin/xpack/license/put")),
        },
        "_features" => cluster(if g(1) == "_reset" { "cluster:admin/features/reset" } else { "cluster:admin/features/get" }),
        "_info" => cluster("cluster:monitor/nodes/stats"),
        "_tasks" => {
            if g(2) == "_cancel" || g(1) == "_cancel" {
                cluster("cluster:admin/tasks/cancel")
            } else if !g(1).is_empty() {
                cluster("cluster:monitor/task/get")
            } else {
                cluster("cluster:monitor/tasks/lists")
            }
        }
        "_data_stream" => match g(1) {
            "_stats" => act("indices:monitor/data_stream/stats", Target::Seg(1)),
            "_migrate" => act("indices:admin/data_stream/migrate", Target::Seg(2)),
            "_promote" => act("indices:admin/data_stream/promote", Target::Seg(2)),
            "_modify" => act("indices:admin/data_stream/modify", Target::AnyIndex),
            _ => match g(2) {
                "_stats" => act("indices:monitor/data_stream/stats", Target::Seg(1)),
                "_lifecycle" => act(by_method(m, "indices:admin/data_stream/lifecycle/get", "indices:admin/data_stream/lifecycle/delete", "indices:admin/data_stream/lifecycle/put"), Target::Seg(1)),
                "_options" => act(if is_read(m) { "indices:admin/data_stream/options/get" } else { "indices:admin/data_stream/options/put" }, Target::Seg(1)),
                _ => act(by_method(m, "indices:admin/data_stream/get", "indices:admin/data_stream/delete", "indices:admin/data_stream/create"), Target::Seg(1)),
            },
        },
        "_index_template" => match g(1) {
            "_simulate_index" => cluster("indices:admin/index_template/simulate_index"),
            "_simulate" => cluster("indices:admin/index_template/simulate"),
            _ => crud(m, "indices:admin/index_template"),
        },
        "_template" => crud(m, "indices:admin/template"),
        "_component_template" => crud(m, "cluster:admin/component_template"),
        "_scripts" => {
            if g(2) == "_execute" {
                cluster("cluster:admin/scripts/painless/execute")
            } else {
                crud(m, "cluster:admin/script")
            }
        }
        "_script_context" => cluster("cluster:admin/script_context/get"),
        "_script_language" => cluster("cluster:admin/script_language/get"),
        "_ingest" => match g(1) {
            "pipeline" => {
                if s.last().copied() == Some("_simulate") {
                    cluster("cluster:admin/ingest/pipeline/simulate")
                } else {
                    crud(m, "cluster:admin/ingest/pipeline")
                }
            }
            "_simulate" => act("indices:data/write/simulate/bulk", Target::AnyIndex),
            "processor" => cluster("cluster:admin/ingest/processor/grok/get"),
            "geoip" | "ip_location" => cluster(if is_read(m) { "cluster:monitor/ingest/geoip/stats" } else { "cluster:admin/ingest/geoip/database/put" }),
            _ => {
                if g(2) == "_simulate" {
                    act("indices:data/write/simulate/bulk", Target::Seg(1))
                } else {
                    cluster("cluster:admin/ingest/pipeline/get")
                }
            }
        },
        "_snapshot" => {
            if s.len() <= 2 {
                cluster(by_method(m, "cluster:admin/repository/get", "cluster:admin/repository/delete", "cluster:admin/repository/put"))
            } else {
                match g(2) {
                    "_verify" => cluster("cluster:admin/repository/verify"),
                    "_cleanup" => cluster("cluster:admin/repository/_cleanup"),
                    "_status" => cluster("cluster:admin/snapshot/status"),
                    "_analyze" => cluster("cluster:admin/repository/analyze"),
                    "_verify_integrity" => cluster("cluster:admin/repository/verify_integrity"),
                    _ => match g(3) {
                        "_restore" => cluster("cluster:admin/snapshot/restore"),
                        "_status" => cluster("cluster:admin/snapshot/status"),
                        "_clone" => cluster("cluster:admin/snapshot/clone"),
                        "_mount" => cluster("cluster:admin/snapshot/mount"),
                        _ => cluster(by_method(m, "cluster:admin/snapshot/get", "cluster:admin/snapshot/delete", "cluster:admin/snapshot/create")),
                    },
                }
            }
        }
        "_slm" => match g(1) {
            "status" => cluster("cluster:admin/slm/status"),
            "stats" => cluster("cluster:admin/slm/stats"),
            "start" => cluster("cluster:admin/slm/start"),
            "stop" => cluster("cluster:admin/slm/stop"),
            "_execute_retention" => cluster("cluster:admin/slm/execute-retention"),
            _ => {
                if g(3) == "_execute" {
                    cluster("cluster:admin/slm/execute")
                } else {
                    crud(m, "cluster:admin/slm")
                }
            }
        },
        "_ilm" => match g(1) {
            "status" => cluster("cluster:admin/ilm/operation_mode/get"),
            "start" => cluster("cluster:admin/ilm/start"),
            "stop" => cluster("cluster:admin/ilm/stop"),
            "move" => cluster("cluster:admin/ilm/_move/post"),
            "migrate_to_data_tiers" => cluster("cluster:admin/migrate_to_data_tiers"),
            _ => crud(m, "cluster:admin/ilm"),
        },
        "_enrich" => {
            if g(1) == "_stats" {
                cluster("cluster:monitor/xpack/enrich/stats")
            } else if g(3) == "_execute" {
                cluster("cluster:admin/xpack/enrich/execute")
            } else {
                crud(m, "cluster:admin/xpack/enrich")
            }
        }
        "_transform" => {
            let last = s.last().copied().unwrap_or("");
            if last == "_stats" {
                cluster("cluster:monitor/transform/stats/get")
            } else if is_read(m) {
                cluster("cluster:monitor/transform/get")
            } else if last.starts_with('_') {
                cluster(format!("cluster:admin/transform/{}", last.trim_start_matches('_')))
            } else if *m == Method::DELETE {
                cluster("cluster:admin/transform/delete")
            } else {
                cluster("cluster:admin/transform/put")
            }
        }
        "_ml" => cluster(if is_read(m) { format!("cluster:monitor/xpack/ml/{}/get", g(1)) } else { format!("cluster:admin/xpack/ml/{}", g(1)) }),
        "_watcher" => cluster(if is_read(m) { format!("cluster:monitor/xpack/watcher/{}", g(1)) } else { format!("cluster:admin/xpack/watcher/{}", g(1)) }),
        "_rollup" => cluster(if is_read(m) { "cluster:monitor/xpack/rollup/get".to_string() } else { format!("cluster:admin/xpack/rollup/{}", g(1)) }),
        "_ccr" => cluster(match (g(1), g(3)) {
            ("stats", _) => "cluster:monitor/ccr/stats",
            ("auto_follow", "pause" | "resume") => "cluster:admin/xpack/ccr/auto_follow_pattern/activate",
            ("auto_follow", _) => match *m {
                Method::GET | Method::HEAD => "cluster:admin/xpack/ccr/auto_follow_pattern/get",
                Method::DELETE => "cluster:admin/xpack/ccr/auto_follow_pattern/delete",
                _ => "cluster:admin/xpack/ccr/auto_follow_pattern/put",
            },
            _ => "cluster:admin/xpack/ccr/unknown",
        }),
        "_logstash" => crud(m, "cluster:admin/logstash/pipeline"),
        "_autoscaling" => cluster(format!("cluster:admin/autoscaling/{}", g(1))),
        "_application" => match g(1) {
            "analytics" => cluster(if g(3) == "event" {
                "cluster:admin/xpack/application/analytics/post_event"
            } else {
                by_method_str(m, "cluster:admin/xpack/application/analytics/get", "cluster:admin/xpack/application/analytics/delete", "cluster:admin/xpack/application/analytics/put")
            }),
            _ => match g(3) {
                "_search" => act("indices:data/read/xpack/application/search_application/search", Target::Seg(2)),
                "_render_query" => cluster("cluster:admin/xpack/application/search_application/render_query"),
                _ if g(2).is_empty() => cluster("cluster:admin/xpack/application/search_application/list"),
                _ => crud(m, "cluster:admin/xpack/application/search_application"),
            },
        },
        "_query_rules" => cluster(match (g(1), g(2)) {
            ("", _) => "cluster:admin/xpack/query_rules/list",
            (_, "_test") => "cluster:admin/xpack/query_rules/test",
            (_, "_rule") => by_method_str(m, "cluster:admin/xpack/query_rules/rule/get", "cluster:admin/xpack/query_rules/rule/delete", "cluster:admin/xpack/query_rules/rule/put"),
            _ => by_method_str(m, "cluster:admin/xpack/query_rules/get", "cluster:admin/xpack/query_rules/delete", "cluster:admin/xpack/query_rules/put"),
        }),
        "_connector" => connector(m, &s),
        "_synonyms" => crud(m, "cluster:admin/synonyms"),
        "_monitoring" => cluster("cluster:admin/xpack/monitoring/bulk"),
        "_inference" if g(1) == "_region_policy" => match *m {
            Method::GET | Method::HEAD => cluster("cluster:monitor/xpack/inference/region_policy/get"),
            Method::DELETE => cluster("cluster:admin/xpack/inference/region_policy/delete"),
            _ => cluster("cluster:admin/xpack/inference/region_policy/put"),
        },
        "_inference" => match *m {
            Method::GET | Method::HEAD => cluster("cluster:monitor/xpack/inference/get"),
            Method::DELETE => cluster("cluster:admin/xpack/inference/delete"),
            Method::PUT => cluster("cluster:admin/xpack/inference/put"),
            _ => cluster("cluster:monitor/xpack/inference/post"),
        },
        "_text_structure" => cluster(format!("cluster:monitor/text_structure/{}", g(1).trim_start_matches('_'))),
        "_health_report" => cluster("cluster:monitor/health_api"),
        "_migration" => cluster(if g(1) == "deprecations" { "cluster:admin/xpack/deprecation/info".to_string() } else { format!("cluster:admin/migration/{}", if is_read(m) { "get_system_feature" } else { "post_system_feature" }) }),
        "_remote" => cluster("cluster:monitor/remote/info"),
        "_ssl" => cluster("cluster:monitor/xpack/ssl/certificates/get"),
        "_capabilities" => cluster("cluster:monitor/nodes/capabilities"),
        "_searchable_snapshots" => cluster("cluster:monitor/xpack/searchable_snapshots/stats"),
        "_dangling" => cluster(if is_read(m) { "cluster:admin/indices/dangling/list" } else { "cluster:admin/indices/dangling/import" }),
        "_internal" => cluster(format!("cluster:admin/internal/{}", g(1))),
        "_fleet" if g(1) == "secret" => cluster(match *m {
            Method::GET | Method::HEAD => "cluster:admin/fleet/secrets/get",
            Method::DELETE => "cluster:admin/fleet/secrets/delete",
            _ => "cluster:admin/fleet/secrets/post",
        }),
        // `RestFleetMultiSearchAction` runs `TransportMultiSearchAction`:
        // every item is authorized as a search on its own targets.
        "_fleet" if g(1) == "_fleet_msearch" => act("indices:data/read/msearch", Target::Msearch),
        "_fleet" => act("indices:data/read/search", Target::AnyIndex),
        other => cluster(if is_read(m) { format!("cluster:monitor/{}", other.trim_start_matches('_')) } else { format!("cluster:admin/{}", other.trim_start_matches('_')) }),
    }
}

fn classify_index(req: &Req, s: &[&str]) -> Act {
    let m = &req.method;
    let g = |i: usize| s.get(i).copied().unwrap_or("");
    let idx = |a: &str| act(a, Target::Seg(0));
    let api = g(1);
    match api {
        "" => idx(&by_method(m, "indices:admin/get", "indices:admin/delete", "indices:admin/create")),
        "_doc" | "_create" | "_update" | "_source" => {
            let has_id = !g(2).is_empty();
            if api == "_source" || (api == "_doc" && is_read(m)) {
                return idx("indices:data/read/get");
            }
            if api == "_doc" && *m == Method::DELETE {
                let mut a = idx("indices:data/write/delete");
                a.item = Some("indices:data/write/delete".into());
                return a;
            }
            if api == "_update" {
                let mut a = idx("indices:data/write/update");
                a.item = Some("indices:data/write/update".into());
                a.auto_create = true;
                return a;
            }
            // REST: an auto-id index defaults to `op_type=create` unless
            // `op_type=index` is explicit.
            let create = api == "_create" || req.param("op_type") == Some("create") || (!has_id && req.param("op_type") != Some("index"));
            let mut a = idx("indices:data/write/index");
            a.item = Some(if create { "indices:data/write/index:op_type/create" } else { "indices:data/write/index:op_type/index" }.into());
            a.auto_create = true;
            a
        }
        "_bulk" => act("indices:data/write/bulk", Target::Bulk),
        "_mget" => act("indices:data/read/mget", Target::Mget),
        "_search" => idx(if g(2) == "template" { "indices:data/read/search/template" } else { "indices:data/read/search" }),
        "_msearch" => act("indices:data/read/msearch", Target::Msearch),
        "_count" => idx("indices:data/read/search"),
        "_pit" => idx("indices:data/read/open_point_in_time"),
        "_explain" => idx("indices:data/read/explain"),
        "_termvectors" => idx("indices:data/read/tv"),
        "_mtermvectors" => idx("indices:data/read/mtv"),
        "_field_caps" => idx("indices:data/read/field_caps"),
        "_terms_enum" => idx("indices:data/read/xpack/termsenum/list"),
        "_validate" => idx("indices:admin/validate/query"),
        "_eql" => idx("indices:data/read/eql"),
        "_rank_eval" => idx("indices:data/read/rank_eval"),
        "_async_search" => idx("indices:data/read/async_search/submit"),
        "_knn_search" => idx("indices:data/read/knn_search"),
        "_update_by_query" => idx("indices:data/write/update/byquery"),
        "_delete_by_query" => idx("indices:data/write/delete/byquery"),
        "_mapping" | "_mappings" => idx(if g(2) == "field" { "indices:admin/mappings/fields/get" } else if is_read(m) { "indices:admin/mappings/get" } else { "indices:admin/mapping/put" }),
        "_settings" => idx(if is_read(m) { "indices:monitor/settings/get" } else { "indices:admin/settings/update" }),
        "_alias" | "_aliases" => {
            if is_read(m) {
                idx("indices:admin/aliases/get")
            } else {
                let mut a = idx("indices:admin/aliases");
                a.alias_seg = Some(2);
                a
            }
        }
        "_refresh" => idx("indices:admin/refresh"),
        "_flush" => idx("indices:admin/flush"),
        "_forcemerge" => idx("indices:admin/forcemerge"),
        "_cache" => idx("indices:admin/cache/clear"),
        "_analyze" => idx("indices:admin/analyze"),
        "_stats" => idx("indices:monitor/stats"),
        "_segments" => idx("indices:monitor/segments"),
        "_recovery" => idx("indices:monitor/recovery"),
        "_shard_stores" => idx("indices:monitor/shard_stores"),
        "_disk_usage" => idx("indices:admin/analyze_disk_usage"),
        "_field_usage_stats" => idx("indices:monitor/field_usage_stats"),
        "_open" => idx("indices:admin/open"),
        "_close" => idx("indices:admin/close"),
        "_block" => idx(if *m == Method::DELETE { "indices:admin/block/remove" } else { "indices:admin/block/add" }),
        "_rollover" => idx("indices:admin/rollover"),
        "_shrink" | "_split" | "_clone" => idx("indices:admin/resize"),
        "_search_shards" => idx("indices:admin/shards/search_shards"),
        "_ilm" => idx(match g(2) {
            "remove" => "indices:admin/ilm/remove_policy",
            "retry" => "indices:admin/ilm/retry",
            _ => "indices:admin/ilm/explain",
        }),
        "_lifecycle" => idx("indices:admin/data_stream/lifecycle/explain"),
        "_reload_search_analyzers" => idx("indices:admin/reload_analyzers"),
        "_downsample" => idx("indices:admin/xpack/downsample"),
        "_unfreeze" => idx("indices:admin/freeze"),
        // Fleet search runs `TransportSearchAction`, fleet msearch
        // `TransportMultiSearchAction`; only global checkpoints is a
        // monitor action.
        "_fleet" => match g(2) {
            "_fleet_search" => idx("indices:data/read/search"),
            "_fleet_msearch" => act("indices:data/read/msearch", Target::Msearch),
            _ => idx("indices:monitor/fleet/global_checkpoints"),
        },
        "_ccr" => match g(2) {
            "follow" => idx("indices:admin/xpack/ccr/put_follow"),
            "unfollow" => idx("indices:admin/xpack/ccr/unfollow"),
            "forget_follower" => idx("indices:admin/xpack/ccr/forget_follower"),
            "pause_follow" => cluster("cluster:admin/xpack/ccr/pause_follow"),
            "resume_follow" => cluster("cluster:admin/xpack/ccr/resume_follow"),
            "info" => cluster("cluster:monitor/ccr/follow_info"),
            "stats" => cluster("cluster:monitor/ccr/follow_stats"),
            other => idx(&format!("indices:admin/xpack/ccr/{other}")),
        },
        "_migration" => idx("indices:admin/migration/reindex_status"),
        other if other.starts_with('_') => idx(&if is_read(m) { format!("indices:monitor/{}", other.trim_start_matches('_')) } else { format!("indices:admin/{}", other.trim_start_matches('_')) }),
        // Legacy `/{index}/{type}/{id}` shapes.
        _ => idx(if is_read(m) { "indices:data/read/get" } else { "indices:data/write/index" }),
    }
}
