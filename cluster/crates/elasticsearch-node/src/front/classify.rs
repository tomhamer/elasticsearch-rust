//! Which node must execute a REST request.
//!
//! Every node mirrors all cluster metadata into its local engine, but shard
//! data only exists where the routing table put a copy. So the cluster
//! front sorts each request into one of a few execution strategies and the
//! engine's own REST layer does the actual work wherever it ends up:
//!
//! | Route | Executed on |
//! |---|---|
//! | `Cluster` | here, by the node crate (`_cluster/*`, `_nodes/*`, cluster `_cat`s, `/`) |
//! | `Local` | here, against the metadata mirror (mappings, settings, aliases, templates, `_analyze`, ...) |
//! | `Master` | the elected master, which then publishes the resulting metadata |
//! | `Write` | the node holding the primary of the (single) target index |
//! | `Bulk` | split per primary node, items merged back in request order |
//! | `Read` | a node holding an active copy of every target index |
//! | `Broadcast` | every node holding copies of the targets; responses merged |
//! | `Context` | the node that owns a scroll / PIT / async-search id |
//!
//! The table of paths mirrors `elasticsearch_http::dispatch`; unknown paths
//! fall back to `Local` with reads and `Master` with writes, the safe choice
//! for metadata-style APIs.

use axum::http::Method;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Route {
    Cluster(ClusterApi),
    Local,
    Master,
    /// Single-index write; `index` may be an alias/data stream/date math.
    Write { index: String },
    Bulk { default_index: Option<String> },
    /// `targets` is the raw index expression (empty = all indices).
    Read { targets: String },
    Broadcast { targets: String, op: BroadcastOp },
    Context(ContextKind),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClusterApi {
    Root,
    Health,
    State,
    Stats,
    Settings,
    PendingTasks,
    Reroute,
    AllocationExplain,
    VotingConfigExclusions,
    NodesInfo,
    NodesStats,
    NodesHotThreads,
    NodesUsage,
    NodeShutdown,
    CatNodes,
    CatMaster,
    CatHealth,
    CatShards,
    CatAllocation,
    CatRecovery,
    CatIndices,
    CatCount,
    CatSegments,
    CatPendingTasks,
    CatNodeattrs,
    CatSnapshots,
    CatRepositories,
    CatThreadPool,
    CatPlugins,
    Snapshot,
    Slm,
    Recovery,
    RemoteInfo,
    DesiredBalance,
    /// `_internal/desired_nodes` (rest/desired_nodes.rs).
    DesiredNodes,
    /// `_autoscaling/*` (rest/autoscaling.rs).
    Autoscaling,
    /// `_dangling[/{uuid}]` (rest/dangling.rs).
    Dangling,
    PrevalidateNodeRemoval,
    /// `GET /_capabilities` (rest/capabilities.rs).
    Capabilities,
    /// `_field_caps`: per-index fan-out merged on the coordinating node.
    FieldCaps,
    /// `_segments`: per-copy segment listing from every node.
    Segments,
    /// `_cache/clear`.
    ClearCache,
    /// `_resolve/cluster`.
    ResolveCluster,
    /// `_nodes/reload_secure_settings`.
    ReloadSecureSettings,
    /// The deprecated `_knn_search` endpoint (runs as a `knn` search).
    KnnSearch,
    /// `_tasks` list/get/cancel, merged across nodes.
    Tasks,
    /// `_cat/tasks`, merged across nodes.
    CatTasks,
    CatFielddata,
    /// `_reindex|_update_by_query|_delete_by_query/{id}/_rethrottle`.
    TaskRethrottle,
    ShardStores,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BroadcastOp {
    Refresh,
    Flush,
    ForceMerge,
    ClearCache,
    Stats,
    Segments,
    ShardStores,
    /// `POST /{index}/_disk_usage`: runs on a copy of every shard.
    DiskUsage,
    /// `GET /{index}/_field_usage_stats`: runs on every node holding a copy.
    FieldUsage,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContextKind {
    Scroll,
    ClearScroll,
    ClosePit,
    AsyncSearch,
}

fn is_read_method(m: &Method) -> bool {
    *m == Method::GET || *m == Method::HEAD
}

/// Classify a request by method and decoded path segments.
pub fn classify(method: &Method, segs: &[&str]) -> Route {
    use ClusterApi as C;
    let read = is_read_method(method);
    let Some(&first) = segs.first() else {
        return Route::Cluster(C::Root);
    };
    let second = segs.get(1).copied().unwrap_or("");
    // `_all` is an index expression, not an API namespace.
    if first.starts_with('_') && first != "_all" {
        return match first {
            "_cluster" => match second {
                "health" => Route::Cluster(C::Health),
                "state" => Route::Cluster(C::State),
                "stats" => Route::Cluster(C::Stats),
                "settings" => Route::Cluster(C::Settings),
                "pending_tasks" => Route::Cluster(C::PendingTasks),
                "reroute" => Route::Cluster(C::Reroute),
                "allocation" => Route::Cluster(C::AllocationExplain),
                "voting_config_exclusions" => Route::Cluster(C::VotingConfigExclusions),
                _ if read => Route::Local,
                _ => Route::Master,
            },
            "_nodes" => nodes_route(segs),
            "_remote" => Route::Cluster(C::RemoteInfo),
            "_capabilities" => Route::Cluster(C::Capabilities),
            "_autoscaling" => Route::Cluster(C::Autoscaling),
            "_dangling" => Route::Cluster(C::Dangling),
            "_internal" if second == "desired_balance" => Route::Cluster(C::DesiredBalance),
            "_internal" if second == "desired_nodes" => Route::Cluster(C::DesiredNodes),
            "_internal" if second == "prevalidate_node_removal" => Route::Cluster(C::PrevalidateNodeRemoval),
            "_recovery" => Route::Cluster(C::Recovery),
            "_cat" => cat_route(segs),
            "_snapshot" => Route::Cluster(C::Snapshot),
            "_slm" => Route::Cluster(C::Slm),
            "_search" if second == "scroll" => {
                if *method == Method::DELETE { Route::Context(ContextKind::ClearScroll) } else { Route::Context(ContextKind::Scroll) }
            }
            "_pit" if *method == Method::DELETE => Route::Context(ContextKind::ClosePit),
            // `GET|DELETE /_async_search/{id}`, `GET /_async_search/status/{id}`.
            "_async_search" if !second.is_empty() => Route::Context(ContextKind::AsyncSearch),
            "_field_caps" => Route::Cluster(C::FieldCaps),
            "_knn_search" => Route::Cluster(C::KnnSearch),
            "_resolve" if second == "cluster" => Route::Cluster(C::ResolveCluster),
            "_search" | "_msearch" | "_count" | "_mget" | "_mtermvectors" | "_pit" | "_async_search" | "_search_shards" | "_validate" | "_terms_enum" | "_rank_eval" => {
                Route::Read { targets: String::new() }
            }
            "_bulk" => Route::Bulk { default_index: None },
            "_refresh" => Route::Broadcast { targets: String::new(), op: BroadcastOp::Refresh },
            "_flush" => Route::Broadcast { targets: String::new(), op: BroadcastOp::Flush },
            "_forcemerge" => Route::Broadcast { targets: String::new(), op: BroadcastOp::ForceMerge },
            "_cache" => Route::Cluster(C::ClearCache),
            "_stats" => Route::Broadcast { targets: String::new(), op: BroadcastOp::Stats },
            "_segments" => Route::Cluster(C::Segments),
            "_shard_stores" => Route::Cluster(C::ShardStores),
            // Reindex / by-query run where the data is; see `Write`.
            "_reindex" | "_update_by_query" | "_delete_by_query" if segs.get(2) == Some(&"_rethrottle") => {
                Route::Cluster(C::TaskRethrottle)
            }
            "_reindex" | "_update_by_query" | "_delete_by_query" => Route::Master,
            "_tasks" => Route::Cluster(C::Tasks),
            // Transform configs, state and running tasks live on the
            // master's engine: reads go there too.
            "_transform" => Route::Master,
            // EQL runs (and keeps async results) on the master's engine.
            "_eql" => Route::Master,
        "_ccr" => Route::Master,
            // Watches, their cache and the scheduler live on the master's engine.
            "_watcher" => Route::Master,
            // Auto-follow patterns and follower tasks live on the master's engine.
            "_ccr" => Route::Master,
            // The profiling status installs the plugin's templates, ILM
            // policy, indices and data streams (ES runs that registry on the
            // master): the master's engine, then published.
            "_profiling" => Route::Master,
            // Kibana enrollment (a GET) creates a service account token.
            "_security" if super::is_kibana_enrollment(segs) => Route::Master,
            _ if read => Route::Local,
            // Templates, aliases, data streams, ingest pipelines, ILM,
            // stored scripts, `_security`, ...: cluster metadata.
            _ => Route::Master,
        };
    }
    // `/{index}/...`
    let index = first.to_string();
    match second {
        "" => {
            if read { Route::Local } else { Route::Master }
        }
        "_doc" | "_create" | "_update" => {
            if read { Route::Read { targets: index } } else { Route::Write { index } }
        }
        "_field_caps" => Route::Cluster(C::FieldCaps),
        "_knn_search" => Route::Cluster(C::KnnSearch),
        "_source" | "_explain" | "_termvectors" | "_mtermvectors" | "_mget" | "_search" | "_msearch" | "_count" | "_pit" | "_async_search" | "_search_shards" | "_validate" | "_terms_enum" | "_rank_eval" => {
            Route::Read { targets: index }
        }
        "_bulk" => Route::Bulk { default_index: Some(index) },
        "_update_by_query" | "_delete_by_query" => Route::Write { index },
        "_refresh" => Route::Broadcast { targets: index, op: BroadcastOp::Refresh },
        "_flush" => Route::Broadcast { targets: index, op: BroadcastOp::Flush },
        "_forcemerge" => Route::Broadcast { targets: index, op: BroadcastOp::ForceMerge },
        "_cache" => Route::Cluster(C::ClearCache),
        "_stats" => Route::Broadcast { targets: index, op: BroadcastOp::Stats },
        "_segments" => Route::Cluster(C::Segments),
        "_shard_stores" => Route::Cluster(C::ShardStores),
        "_recovery" => Route::Cluster(C::Recovery),
        "_disk_usage" if !read => Route::Broadcast { targets: index, op: BroadcastOp::DiskUsage },
        "_eql" => Route::Master,
        "_ccr" => Route::Master,
        "_field_usage_stats" if read => Route::Broadcast { targets: index, op: BroadcastOp::FieldUsage },
        // `_analyze`, `_mapping`, `_settings`, `_alias`, ... GETs read the
        // mirror; changes (`_close`, `_open`, `_rollover`, `_mapping` PUT,
        // `_block`, `_shrink`, ...) are metadata.
        _ if read => Route::Local,
        "_analyze" => Route::Local,
        _ => Route::Master,
    }
}

fn nodes_route(segs: &[&str]) -> Route {
    use ClusterApi as C;
    let has = |s: &str| segs.iter().skip(1).any(|x| *x == s);
    if has("stats") {
        Route::Cluster(C::NodesStats)
    } else if has("hot_threads") {
        Route::Cluster(C::NodesHotThreads)
    } else if has("usage") {
        Route::Cluster(C::NodesUsage)
    } else if has("shutdown") {
        Route::Cluster(C::NodeShutdown)
    } else if has("reload_secure_settings") {
        Route::Cluster(C::ReloadSecureSettings)
    } else {
        Route::Cluster(C::NodesInfo)
    }
}

fn cat_route(segs: &[&str]) -> Route {
    use ClusterApi as C;
    match segs.get(1).copied().unwrap_or("") {
        "nodes" => Route::Cluster(C::CatNodes),
        "master" => Route::Cluster(C::CatMaster),
        "health" => Route::Cluster(C::CatHealth),
        "shards" => Route::Cluster(C::CatShards),
        "allocation" => Route::Cluster(C::CatAllocation),
        "recovery" => Route::Cluster(C::CatRecovery),
        "indices" => Route::Cluster(C::CatIndices),
        "count" => Route::Cluster(C::CatCount),
        "segments" => Route::Cluster(C::CatSegments),
        "pending_tasks" => Route::Cluster(C::CatPendingTasks),
        "nodeattrs" => Route::Cluster(C::CatNodeattrs),
        "snapshots" => Route::Cluster(C::CatSnapshots),
        "repositories" => Route::Cluster(C::CatRepositories),
        "thread_pool" => Route::Cluster(C::CatThreadPool),
        "plugins" => Route::Cluster(C::CatPlugins),
        "tasks" => Route::Cluster(C::CatTasks),
        "fielddata" => Route::Cluster(C::CatFielddata),
        "transforms" => Route::Master,
        // aliases, templates, fielddata, ml, ...: the mirror knows.
        _ => Route::Local,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(m: Method, p: &str) -> Route {
        let segs: Vec<&str> = p.split('/').filter(|s| !s.is_empty()).collect();
        classify(&m, &segs)
    }

    #[test]
    fn enrollment_routes() {
        assert_eq!(c(Method::GET, "/_security/enroll/node"), Route::Local);
        assert_eq!(c(Method::GET, "/_security/enroll/kibana"), Route::Master);
    }

    #[test]
    fn profiling_runs_on_the_master() {
        assert_eq!(c(Method::GET, "/_profiling/status"), Route::Master);
        assert_eq!(c(Method::POST, "/_profiling/stacktraces"), Route::Master);
    }

    #[test]
    fn tasks_routes() {
        assert_eq!(c(Method::GET, "/_tasks"), Route::Cluster(ClusterApi::Tasks));
        assert_eq!(c(Method::GET, "/_tasks/n1:5"), Route::Cluster(ClusterApi::Tasks));
        assert_eq!(c(Method::POST, "/_tasks/_cancel"), Route::Cluster(ClusterApi::Tasks));
        assert_eq!(c(Method::POST, "/_tasks/n1:5/_cancel"), Route::Cluster(ClusterApi::Tasks));
        assert_eq!(c(Method::GET, "/_cat/tasks"), Route::Cluster(ClusterApi::CatTasks));
        // Fielddata lives on the nodes holding the copies: gathered from all.
        assert_eq!(c(Method::GET, "/_cat/fielddata"), Route::Cluster(ClusterApi::CatFielddata));
        assert_eq!(c(Method::GET, "/_cat/fielddata/foo"), Route::Cluster(ClusterApi::CatFielddata));
        assert_eq!(c(Method::POST, "/_update_by_query/n1:5/_rethrottle"), Route::Cluster(ClusterApi::TaskRethrottle));
        assert_eq!(c(Method::POST, "/_reindex/n1:5/_rethrottle"), Route::Cluster(ClusterApi::TaskRethrottle));
        assert_eq!(c(Method::POST, "/_reindex"), Route::Master);
    }

    #[test]
    fn cluster_answered_index_apis() {
        assert_eq!(c(Method::GET, "/_field_caps"), Route::Cluster(ClusterApi::FieldCaps));
        assert_eq!(c(Method::POST, "/a,b/_field_caps"), Route::Cluster(ClusterApi::FieldCaps));
        assert_eq!(c(Method::GET, "/_segments"), Route::Cluster(ClusterApi::Segments));
        assert_eq!(c(Method::GET, "/logs/_segments"), Route::Cluster(ClusterApi::Segments));
        assert_eq!(c(Method::POST, "/_cache/clear"), Route::Cluster(ClusterApi::ClearCache));
        assert_eq!(c(Method::POST, "/logs/_cache/clear"), Route::Cluster(ClusterApi::ClearCache));
        assert_eq!(c(Method::GET, "/_resolve/cluster/logs*"), Route::Cluster(ClusterApi::ResolveCluster));
        assert_eq!(c(Method::POST, "/_nodes/reload_secure_settings"), Route::Cluster(ClusterApi::ReloadSecureSettings));
        assert_eq!(c(Method::POST, "/_nodes/n1/reload_secure_settings"), Route::Cluster(ClusterApi::ReloadSecureSettings));
        assert_eq!(c(Method::POST, "/logs/_knn_search"), Route::Cluster(ClusterApi::KnnSearch));
        assert_eq!(c(Method::GET, "/logs/_shard_stores"), Route::Cluster(ClusterApi::ShardStores));
        assert_eq!(c(Method::GET, "/_shard_stores"), Route::Cluster(ClusterApi::ShardStores));
    }

    #[test]
    fn transforms_run_on_the_master() {
        assert_eq!(c(Method::GET, "/_transform/t"), Route::Master);
        assert_eq!(c(Method::GET, "/_transform/t/_stats"), Route::Master);
        assert_eq!(c(Method::POST, "/_transform/_preview"), Route::Master);
        assert_eq!(c(Method::GET, "/_cat/transforms"), Route::Master);
    }

    #[test]
    fn watcher_runs_on_the_master() {
        assert_eq!(c(Method::GET, "/_watcher/watch/w"), Route::Master);
        assert_eq!(c(Method::GET, "/_watcher/_query/watches"), Route::Master);
        assert_eq!(c(Method::GET, "/_watcher/stats"), Route::Master);
    }

    #[test]
    fn ccr_runs_on_the_master() {
        assert_eq!(c(Method::GET, "/_ccr/auto_follow/p"), Route::Master);
        assert_eq!(c(Method::GET, "/_ccr/stats"), Route::Master);
        assert_eq!(c(Method::GET, "/bar/_ccr/info"), Route::Master);
        assert_eq!(c(Method::GET, "/bar/_ccr/stats"), Route::Master);
    }

    #[test]
    fn eql_runs_on_the_master() {
        assert_eq!(c(Method::GET, "/logs/_eql/search"), Route::Master);
        assert_eq!(c(Method::POST, "/logs/_eql/search"), Route::Master);
        assert_eq!(c(Method::GET, "/_eql/search/status/abc"), Route::Master);
        assert_eq!(c(Method::DELETE, "/_eql/search/abc"), Route::Master);
    }

    #[test]
    fn documents() {
        assert_eq!(c(Method::PUT, "/logs/_doc/1"), Route::Write { index: "logs".into() });
        assert_eq!(c(Method::POST, "/logs/_doc"), Route::Write { index: "logs".into() });
        assert_eq!(c(Method::DELETE, "/logs/_doc/1"), Route::Write { index: "logs".into() });
        assert_eq!(c(Method::POST, "/logs/_update/1"), Route::Write { index: "logs".into() });
        assert_eq!(c(Method::GET, "/logs/_doc/1"), Route::Read { targets: "logs".into() });
        assert_eq!(c(Method::HEAD, "/logs/_doc/1"), Route::Read { targets: "logs".into() });
        assert_eq!(c(Method::POST, "/_bulk"), Route::Bulk { default_index: None });
        assert_eq!(c(Method::PUT, "/logs/_bulk"), Route::Bulk { default_index: Some("logs".into()) });
        assert_eq!(c(Method::POST, "/logs/_delete_by_query"), Route::Write { index: "logs".into() });
    }

    #[test]
    fn search_and_contexts() {
        assert_eq!(c(Method::POST, "/a,b/_search"), Route::Read { targets: "a,b".into() });
        assert_eq!(c(Method::GET, "/_search"), Route::Read { targets: "".into() });
        assert_eq!(c(Method::POST, "/_search/scroll"), Route::Context(ContextKind::Scroll));
        assert_eq!(c(Method::DELETE, "/_search/scroll"), Route::Context(ContextKind::ClearScroll));
        assert_eq!(c(Method::DELETE, "/_pit"), Route::Context(ContextKind::ClosePit));
        assert_eq!(c(Method::POST, "/i/_pit"), Route::Read { targets: "i".into() });
        assert_eq!(c(Method::GET, "/_async_search/abc"), Route::Context(ContextKind::AsyncSearch));
        assert_eq!(c(Method::POST, "/i/_async_search"), Route::Read { targets: "i".into() });
    }

    #[test]
    fn metadata_and_cluster() {
        assert_eq!(c(Method::PUT, "/logs"), Route::Master);
        assert_eq!(c(Method::DELETE, "/logs"), Route::Master);
        assert_eq!(c(Method::GET, "/logs"), Route::Local);
        assert_eq!(c(Method::PUT, "/logs/_mapping"), Route::Master);
        assert_eq!(c(Method::GET, "/logs/_mapping"), Route::Local);
        assert_eq!(c(Method::POST, "/logs/_close"), Route::Master);
        assert_eq!(c(Method::POST, "/logs/_analyze"), Route::Local);
        assert_eq!(c(Method::PUT, "/_index_template/t"), Route::Master);
        assert_eq!(c(Method::GET, "/_index_template/t"), Route::Local);
        assert_eq!(c(Method::POST, "/_aliases"), Route::Master);
        assert_eq!(c(Method::GET, "/"), Route::Cluster(ClusterApi::Root));
        assert_eq!(c(Method::GET, "/_cluster/health/logs"), Route::Cluster(ClusterApi::Health));
        assert_eq!(c(Method::PUT, "/_cluster/settings"), Route::Cluster(ClusterApi::Settings));
        assert_eq!(c(Method::GET, "/_nodes/_local/stats/jvm"), Route::Cluster(ClusterApi::NodesStats));
        assert_eq!(c(Method::GET, "/_nodes/http"), Route::Cluster(ClusterApi::NodesInfo));
        assert_eq!(c(Method::GET, "/_cat/shards/logs"), Route::Cluster(ClusterApi::CatShards));
        assert_eq!(c(Method::GET, "/_cat/aliases"), Route::Local);
        assert_eq!(c(Method::PUT, "/_snapshot/repo/snap"), Route::Cluster(ClusterApi::Snapshot));
        assert_eq!(c(Method::POST, "/logs/_refresh"), Route::Broadcast { targets: "logs".into(), op: BroadcastOp::Refresh });
        assert_eq!(c(Method::GET, "/_stats"), Route::Broadcast { targets: "".into(), op: BroadcastOp::Stats });
        assert_eq!(c(Method::GET, "/_internal/desired_balance"), Route::Cluster(ClusterApi::DesiredBalance));
        assert_eq!(c(Method::PUT, "/_internal/desired_nodes/h/1"), Route::Cluster(ClusterApi::DesiredNodes));
        assert_eq!(c(Method::GET, "/_internal/desired_nodes/_latest"), Route::Cluster(ClusterApi::DesiredNodes));
        assert_eq!(c(Method::POST, "/_internal/prevalidate_node_removal"), Route::Cluster(ClusterApi::PrevalidateNodeRemoval));
        assert_eq!(c(Method::GET, "/i/_recovery"), Route::Cluster(ClusterApi::Recovery));
    }
}
