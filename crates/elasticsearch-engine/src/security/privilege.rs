//! Named cluster / index privileges (ES `ClusterPrivilegeResolver`,
//! `IndexPrivilege`), action-name patterns, and the "this action is granted
//! by the ... privileges [..]" ordering (`sortByAccessLevel`).

use std::sync::OnceLock;

/// A named privilege: actions matching any `include` glob and no `exclude` glob.
#[derive(Debug)]
pub struct PrivDef {
    pub name: &'static str,
    pub include: &'static [&'static str],
    pub exclude: &'static [&'static str],
}

const fn p(name: &'static str, include: &'static [&'static str]) -> PrivDef {
    PrivDef { name, include, exclude: &[] }
}

const ALL_CLUSTER: &[&str] = &["cluster:*", "indices:admin/template/*", "indices:admin/index_template/*"];
const SECURITY: &[&str] = &["cluster:admin/xpack/security/*"];

/// Cluster privileges in ES declaration order (62 in 8.19).
pub static CLUSTER_PRIVS: &[PrivDef] = &[
    PrivDef { name: "none", include: &[], exclude: &[] },
    p("all", ALL_CLUSTER),
    p("monitor", &["cluster:monitor/*", "indices:admin/template/get", "cluster:admin/component_template/get", "indices:admin/index_template/get"]),
    p("monitor_connector", &["cluster:admin/xpack/connector/get", "cluster:admin/xpack/connector/list", "cluster:admin/xpack/connector/sync_job/get", "cluster:admin/xpack/connector/sync_job/list"]),
    p("monitor_inference", &["cluster:monitor/xpack/inference*", "cluster:monitor/xpack/ml/trained_models/deployment/infer"]),
    p("monitor_ml", &["cluster:monitor/xpack/ml/*"]),
    p("monitor_text_structure", &["cluster:monitor/text_structure/*"]),
    p("monitor_data_frame_transforms", &["cluster:monitor/data_frame/*", "cluster:monitor/transform/*"]),
    p("monitor_transform", &["cluster:monitor/data_frame/*", "cluster:monitor/transform/*"]),
    p("monitor_watcher", &["cluster:monitor/xpack/watcher/*"]),
    p("monitor_reindex", &["cluster:monitor/reindex/*"]),
    p("monitor_rollup", &["cluster:monitor/xpack/rollup/*"]),
    p("monitor_enrich", &["cluster:monitor/xpack/enrich/*", "cluster:admin/xpack/enrich/get"]),
    p("monitor_esql", &["cluster:monitor/xpack/esql/*"]),
    p("monitor_stats", &["cluster:monitor/stats*"]),
    PrivDef { name: "manage", include: ALL_CLUSTER, exclude: SECURITY },
    p("manage_connector", &["cluster:admin/xpack/connector/*"]),
    p("manage_inference", &["cluster:admin/xpack/inference/*", "cluster:monitor/xpack/inference*", "cluster:admin/xpack/ml/trained_models/deployment/start", "cluster:admin/xpack/ml/trained_models/deployment/stop", "cluster:monitor/xpack/ml/trained_models/deployment/infer"]),
    p("manage_ml", &["cluster:admin/xpack/ml/*", "cluster:monitor/xpack/ml/*"]),
    p("manage_data_frame_transforms", &["cluster:admin/data_frame/*", "cluster:monitor/data_frame/*", "cluster:monitor/transform/*", "cluster:admin/transform/*"]),
    p("manage_transform", &["cluster:admin/data_frame/*", "cluster:monitor/data_frame/*", "cluster:monitor/transform/*", "cluster:admin/transform/*"]),
    p("manage_token", &["cluster:admin/xpack/security/token/*"]),
    p("manage_watcher", &["cluster:admin/xpack/watcher/*", "cluster:monitor/xpack/watcher/*"]),
    p("manage_reindex", &["cluster:admin/reindex/*", "cluster:monitor/reindex/*"]),
    p("manage_index_templates", &["indices:admin/template/*", "indices:admin/index_template/*", "cluster:admin/component_template/*"]),
    p("manage_ingest_pipelines", &["cluster:admin/ingest/pipeline/*"]),
    p("read_pipeline", &["cluster:admin/ingest/pipeline/get", "cluster:admin/ingest/pipeline/simulate"]),
    p("transport_client", &["cluster:monitor/nodes/liveness", "cluster:monitor/state"]),
    p("manage_security", &["cluster:admin/xpack/security/*", "cluster:admin/xpack/security/delegate_pki"]),
    p("read_security", &[
        "cluster:admin/xpack/security/api_key/get", "cluster:admin/xpack/security/api_key/query",
        "cluster:admin/xpack/security/privilege/builtin/get", "cluster:admin/xpack/security/privilege/get",
        "cluster:admin/xpack/security/profile/get", "cluster:admin/xpack/security/profile/has_privileges",
        "cluster:admin/xpack/security/profile/suggest", "cluster:admin/xpack/security/role/get",
        "cluster:admin/xpack/security/role/query", "cluster:admin/xpack/security/role_mapping/get",
        "cluster:admin/xpack/security/service_account/get", "cluster:admin/xpack/security/service_account/credential/get*",
        "cluster:admin/xpack/security/user/get", "cluster:admin/xpack/security/user/query",
        "cluster:admin/xpack/security/user/list_privileges", "cluster:admin/xpack/security/user/has_privileges",
        "cluster:admin/xpack/security/settings/get",
    ]),
    p("manage_saml", &["cluster:admin/xpack/security/saml/*", "cluster:admin/xpack/security/token/invalidate", "cluster:admin/xpack/security/token/refresh", "cluster:monitor/xpack/security/saml/metadata"]),
    p("manage_oidc", &["cluster:admin/xpack/security/oidc/*"]),
    p("manage_api_key", &["cluster:admin/xpack/security/api_key/*"]),
    p("grant_api_key", &["cluster:admin/xpack/security/api_key/grant*"]),
    p("clone_api_key", &["cluster:admin/xpack/security/api_key/clone*"]),
    p("manage_service_account", &["cluster:admin/xpack/security/service_account/*"]),
    p("manage_user_profile", &["cluster:admin/xpack/security/profile/*"]),
    p("manage_pipeline", &["cluster:admin/ingest/pipeline/*"]),
    p("manage_rollup", &["cluster:admin/xpack/rollup/*", "cluster:monitor/xpack/rollup/*"]),
    p("manage_autoscaling", &["cluster:admin/autoscaling/*"]),
    p("manage_ccr", &["cluster:admin/xpack/ccr/*", "cluster:monitor/state", "cluster:admin/xpack/security/user/has_privileges"]),
    p("read_ccr", &["cluster:monitor/state", "cluster:admin/xpack/security/user/has_privileges"]),
    p("create_snapshot", &["cluster:admin/snapshot/create", "cluster:admin/snapshot/status*", "cluster:admin/snapshot/get", "cluster:admin/snapshot/status", "cluster:admin/repository/get"]),
    p("monitor_snapshot", &["cluster:admin/snapshot/status*", "cluster:admin/snapshot/get", "cluster:admin/snapshot/status", "cluster:admin/repository/get"]),
    p("manage_ilm", &["cluster:admin/ilm/*"]),
    p("read_ilm", &["cluster:admin/ilm/get", "cluster:admin/ilm/operation_mode/get"]),
    p("manage_slm", &["cluster:admin/slm/*", "cluster:admin/ilm/start", "cluster:admin/ilm/stop", "cluster:admin/ilm/operation_mode/get"]),
    p("read_slm", &["cluster:admin/slm/status", "cluster:admin/slm/get", "cluster:admin/ilm/operation_mode/get"]),
    p("delegate_pki", &["cluster:admin/xpack/security/delegate_pki", "cluster:admin/xpack/security/token/invalidate"]),
    // Conditional in ES (only the caller's own keys); the HTTP layer narrows it.
    p("manage_own_api_key", &["cluster:admin/xpack/security/api_key/create", "cluster:admin/xpack/security/api_key/get", "cluster:admin/xpack/security/api_key/invalidate", "cluster:admin/xpack/security/api_key/query", "cluster:admin/xpack/security/api_key/update", "cluster:admin/xpack/security/api_key/bulk_update"]),
    p("manage_enrich", &["cluster:admin/xpack/enrich/*", "cluster:monitor/xpack/enrich/*"]),
    p("manage_logstash_pipelines", &["cluster:admin/logstash/pipeline/*"]),
    p("read_fleet_secrets", &["cluster:admin/fleet/secrets/get"]),
    p("write_fleet_secrets", &["cluster:admin/fleet/secrets/post", "cluster:admin/fleet/secrets/delete"]),
    p("cancel_task", &["cluster:admin/tasks/cancel*"]),
    p("manage_search_application", &["cluster:admin/xpack/application/search_application/*"]),
    p("manage_search_synonyms", &["cluster:admin/synonyms/*", "cluster:admin/synonyms_sets/*", "cluster:admin/synonym_rules/*"]),
    p("manage_behavioral_analytics", &["cluster:admin/xpack/application/analytics/*"]),
    p("post_behavioral_analytics_event", &["cluster:admin/xpack/application/analytics/post_event"]),
    p("manage_search_query_rules", &["cluster:admin/xpack/query_rules/*"]),
    p("cross_cluster_search", &["cluster:internal/remote_cluster/handshake", "cluster:internal/remote_cluster/nodes", "internal:admin/tasks/ban", "internal:admin/tasks/cancel_child", "cluster:monitor/xpack/info", "cluster:monitor/xpack/enrich/esql/resolve_policy", "cluster:internal:data/read/esql/open_exchange", "cluster:internal:data/read/esql/exchange", "cluster:monitor/stats/remote", "cluster:monitor/stats", "cluster:monitor/stats[n]"]),
    p("cross_cluster_replication", &["cluster:internal/remote_cluster/handshake", "cluster:internal/remote_cluster/nodes", "internal:admin/tasks/ban", "internal:admin/tasks/cancel_child", "cluster:monitor/xpack/info", "cluster:monitor/state"]),
    p("read_connector_secrets", &["cluster:admin/xpack/connector/secret/get"]),
    p("write_connector_secrets", &["cluster:admin/xpack/connector/secret/delete", "cluster:admin/xpack/connector/secret/post", "cluster:admin/xpack/connector/secret/put"]),
    PrivDef { name: "monitor_data_stream_global_retention", include: &[], exclude: &[] },
    PrivDef { name: "manage_data_stream_global_retention", include: &[], exclude: &[] },
];

const MONITOR_IDX: &[&str] = &["indices:monitor/*"];
const MANAGE_IDX: &[&str] = &["indices:monitor/*", "indices:admin/*", "indices:data/read/field_caps*", "indices:data/read/xpack/rollup/get/index/caps*"];

/// Index privileges in ES map order (failure-store ones first, 32 in 9.5.5).
/// `data` = whether it applies to regular (non `::failures`) access.
pub static INDEX_PRIVS: &[(PrivDef, bool)] = &[
    (p("read_failure_store", &["indices:data/read/*", "indices:admin/resolve/index"]), false),
    (p("manage_failure_store", MANAGE_IDX), false),
    (PrivDef { name: "none", include: &[], exclude: &[] }, true),
    (p("all", &["indices:*", "internal:transport/proxy/indices:*"]), true),
    (p("manage", MANAGE_IDX), true),
    (p("create_index", &["indices:admin/create", "indices:admin/auto_create", "indices:admin/data_stream/create"]), true),
    (p("monitor", MONITOR_IDX), true),
    (p("read", &["indices:data/read/*", "indices:admin/resolve/index", "indices:admin/resolve/cluster"]), true),
    (p("index", &["indices:data/write/index*", "indices:data/write/bulk*", "indices:data/write/update*", "indices:data/write/simulate/bulk*"]), true),
    (p("delete", &["indices:data/write/delete*", "indices:data/write/bulk*"]), true),
    (p("write", &["indices:data/write/*", "indices:admin/mapping/auto_put"]), true),
    (p("create", &["indices:data/write/index*", "indices:data/write/bulk*", "indices:data/write/simulate/bulk*"]), true),
    (p("create_doc", &["indices:data/write/index", "indices:data/write/index[*", "indices:data/write/index:op_type/create", "indices:data/write/bulk*", "indices:data/write/simulate/bulk*"]), true),
    (p("delete_index", &["indices:admin/delete", "indices:admin/data_stream/delete"]), true),
    (p("view_index_metadata", &[
        "indices:admin/aliases/get", "indices:admin/get", "indices:admin/mappings/fields/get*", "indices:admin/mappings/get",
        "indices:admin/shards/search_shards", "indices:admin/search/search_shards", "indices:admin/validate/query*",
        "indices:monitor/settings/get", "indices:admin/ilm/explain", "indices:admin/data_stream/lifecycle/get",
        "indices:admin/data_stream/lifecycle/explain", "indices:admin/data_stream/get", "indices:admin/resolve/index",
        "indices:admin/resolve/cluster", "indices:data/read/field_caps*", "indices:data/read/xpack/rollup/get/index/caps*",
        "indices:monitor/transform/checkpoint*", "indices:monitor/get/metering/stats", "indices:admin/get/metering/stats",
    ]), true),
    (p("read_cross_cluster", &["internal:transport/proxy/indices:data/read/*", "indices:admin/shards/search_shards", "indices:admin/search/search_shards", "indices:admin/resolve/cluster", "indices:data/read/esql", "indices:data/read/esql/compute"]), true),
    (p("manage_follow_index", &["indices:admin/xpack/ccr/put_follow", "indices:admin/xpack/ccr/unfollow", "indices:admin/close*", "indices:admin/data_stream/promote", "indices:admin/rollover"]), true),
    (p("manage_leader_index", &["indices:admin/xpack/ccr/forget_follower*"]), true),
    (p("manage_ilm", &["indices:admin/ilm/*"]), true),
    (p("manage_data_stream_lifecycle", &["indices:admin/data_stream/lifecycle/*"]), true),
    (p("maintenance", &["indices:admin/refresh*", "indices:admin/flush*", "indices:admin/synced_flush", "indices:admin/forcemerge*"]), true),
    (p("auto_configure", &["indices:admin/mapping/auto_put", "indices:admin/auto_create"]), true),
    (p("cross_cluster_replication", &["indices:data/read/xpack/ccr/shard_changes*", "indices:monitor/stats*", "indices:admin/seq_no/add_retention_lease*", "indices:admin/seq_no/remove_retention_lease*", "indices:admin/seq_no/renew_retention_lease*"]), true),
    (p("cross_cluster_replication_internal", &["indices:internal/admin/ccr/restore/session/clear*", "indices:internal/admin/ccr/restore/file_chunk/get*", "indices:internal/admin/ccr/restore/session/put*", "internal:transport/proxy/indices:internal/admin/ccr/restore/session/clear*", "internal:transport/proxy/indices:internal/admin/ccr/restore/file_chunk/get*"]), true),
    // ES 9: ES|QL views and datasets.
    (p("manage_view", &["indices:admin/esql/view*"]), true),
    (p("create_view", &["indices:admin/esql/view/put"]), true),
    (p("delete_view", &["indices:admin/esql/view/delete"]), true),
    (p("read_view_metadata", &["indices:admin/esql/view/get"]), true),
    (p("manage_dataset", &["indices:admin/esql/dataset/*"]), true),
    (p("create_dataset", &["indices:admin/esql/dataset/put"]), true),
    (p("delete_dataset", &["indices:admin/esql/dataset/delete"]), true),
    (p("read_dataset_metadata", &["indices:admin/esql/dataset/get"]), true),
];

/// Glob match (`*` any run, `?` one char).
pub fn glob(pat: &str, s: &str) -> bool {
    let (p, t) = (pat.as_bytes(), s.as_bytes());
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut mark) = (usize::MAX, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == b'?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star = pi;
            mark = ti;
            pi += 1;
        } else if star != usize::MAX {
            pi = star + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

/// Is every string matched by glob `q` also matched by glob `p`? (trailing-`*` approximation)
fn glob_covers(p: &str, q: &str) -> bool {
    match p.strip_suffix('*') {
        Some(prefix) if !prefix.contains('*') => q.starts_with(prefix),
        _ => p == q || (!q.contains('*') && glob(p, q)),
    }
}

/// Can globs `a` and `b` match a common string? (trailing-`*` approximation)
fn glob_intersects(a: &str, b: &str) -> bool {
    let pa = a.split('*').next().unwrap_or("");
    let pb = b.split('*').next().unwrap_or("");
    match (a.contains('*'), b.contains('*')) {
        (false, false) => a == b,
        (true, false) => glob(a, b),
        (false, true) => glob(b, a),
        (true, true) => pa.starts_with(pb) || pb.starts_with(pa),
    }
}

impl PrivDef {
    pub fn matches(&self, action: &str) -> bool {
        self.include.iter().any(|p| glob(p, action)) && !self.exclude.iter().any(|p| glob(p, action))
    }
    fn covers(&self, q: &str) -> bool {
        self.include.iter().any(|p| glob_covers(p, q)) && !self.exclude.iter().any(|e| glob_intersects(e, q))
    }
    /// `self` grants everything `other` grants.
    pub fn implies(&self, other: &PrivDef) -> bool {
        other.include.iter().all(|q| self.covers(q))
    }
}

fn sorted_names(defs: &[&PrivDef]) -> Vec<&'static str> {
    let mut v: Vec<(usize, &'static str)> = defs
        .iter()
        .map(|d| (defs.iter().filter(|o| !std::ptr::eq(**o, *d) && d.implies(o)).count(), d.name))
        .collect();
    v.sort();
    v.into_iter().map(|(_, n)| n).collect()
}

/// Cluster privilege names sorted by access level (ES `VALUES` key order).
pub fn cluster_names_sorted() -> &'static [&'static str] {
    static S: OnceLock<Vec<&'static str>> = OnceLock::new();
    S.get_or_init(|| sorted_names(&CLUSTER_PRIVS.iter().collect::<Vec<_>>()))
}

/// Index privilege names: failure-store group sorted, then the rest sorted.
pub fn index_names_sorted() -> &'static [&'static str] {
    static S: OnceLock<Vec<&'static str>> = OnceLock::new();
    S.get_or_init(|| {
        let fs: Vec<&PrivDef> = INDEX_PRIVS.iter().filter(|(_, d)| !*d).map(|(p, _)| p).collect();
        let rest: Vec<&PrivDef> = INDEX_PRIVS.iter().filter(|(_, d)| *d).map(|(p, _)| p).collect();
        let mut out = sorted_names(&fs);
        out.extend(sorted_names(&rest));
        out
    })
}

pub fn cluster_priv(name: &str) -> Option<&'static PrivDef> {
    CLUSTER_PRIVS.iter().find(|d| d.name == name)
}

pub fn index_priv(name: &str) -> Option<&'static PrivDef> {
    INDEX_PRIVS.iter().find(|(d, _)| d.name == name).map(|(d, _)| d)
}

pub fn is_cluster_action(a: &str) -> bool {
    ALL_CLUSTER.iter().any(|p| glob(p, a))
}

pub fn is_index_action(a: &str) -> bool {
    glob("indices:*", a) || glob("internal:transport/proxy/indices:*", a)
}

/// Cluster privilege names granting `action`, least privileged first.
pub fn cluster_privs_granting(action: &str) -> Vec<&'static str> {
    // manage_own_api_key is conditional (own keys only) and never explains a denial.
    cluster_names_sorted()
        .iter()
        .copied()
        .filter(|n| *n != "manage_own_api_key" && cluster_priv(n).map(|d| d.matches(action)).unwrap_or(false))
        .collect()
}

/// Regular (data-selector) index privilege names granting `action`.
pub fn index_privs_granting(action: &str) -> Vec<&'static str> {
    index_names_sorted()
        .iter()
        .copied()
        .filter(|n| INDEX_PRIVS.iter().any(|(d, data)| *data && d.name == *n && d.matches(action)))
        .collect()
}

/// One cluster privilege entry of a role: a named privilege or an action pattern.
#[derive(Clone, Debug)]
pub enum Grant {
    Named(&'static PrivDef),
    Pattern(String),
}

impl Grant {
    pub fn matches(&self, action: &str) -> bool {
        match self {
            Grant::Named(d) => d.matches(action),
            Grant::Pattern(p) => glob(p, action),
        }
    }
    pub fn name(&self) -> &str {
        match self {
            Grant::Named(d) => d.name,
            Grant::Pattern(p) => p.trim_end_matches('*'),
        }
    }
}

/// Resolve a cluster privilege name (`monitor`, or an action like `cluster:monitor/main`).
pub fn resolve_cluster(name: &str) -> Result<Grant, String> {
    let n = name.to_ascii_lowercase();
    if is_cluster_action(&n) {
        return Ok(Grant::Pattern(format!("{n}*")));
    }
    match cluster_priv(&n) {
        Some(d) => Ok(Grant::Named(d)),
        None => Err(format!(
            "unknown cluster privilege [{n}]. a privilege must be either one of the predefined cluster privilege names [{}] or a pattern over one of the available cluster actions",
            cluster_names_sorted().join(",")
        )),
    }
}

/// Resolve an index privilege name (`read`, or an action like `indices:admin/aliases`).
pub fn resolve_index(name: &str) -> Result<Grant, String> {
    let n = name.to_ascii_lowercase();
    if is_index_action(&n) {
        return Ok(Grant::Pattern(format!("{n}*")));
    }
    match index_priv(&n) {
        Some(d) => Ok(Grant::Named(d)),
        None => Err(format!(
            "unknown index privilege [{n}]. a privilege must be either one of the predefined fixed indices privileges [{}] or a pattern over one of the available index actions",
            index_names_sorted().join(",")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_ordering() {
        assert_eq!(CLUSTER_PRIVS.len(), 66);
        assert_eq!(INDEX_PRIVS.len(), 32);
        assert_eq!(cluster_privs_granting("cluster:admin/xpack/security/api_key/invalidate"), vec!["manage_api_key", "manage_security", "all"]);
        assert_eq!(cluster_privs_granting("cluster:admin/xpack/security/user/delete"), vec!["manage_security", "all"]);
        assert_eq!(index_privs_granting("indices:data/read/search"), vec!["read", "all"]);
        assert_eq!(index_privs_granting("indices:admin/xpack/downsample"), vec!["manage", "all"]);
        assert_eq!(cluster_privs_granting("cluster:admin/xpack/security/api_key/grant"), vec!["grant_api_key", "manage_api_key", "manage_security", "all"]);
    }
}
