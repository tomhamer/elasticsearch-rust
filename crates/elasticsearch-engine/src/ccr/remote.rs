//! Remote cluster connections for CCR (ES `RemoteClusterService`).
//!
//! A remote cluster alias is configured with the `cluster.remote.<alias>.*`
//! settings (cluster settings, transient over persistent, then node
//! settings): `mode` (`sniff`, default, or `proxy`), `seeds` (sniff) or
//! `proxy_address` (proxy). An alias without addresses does not exist
//! (`no_such_remote_cluster_exception`).
//!
//! The leader side of every CCR operation goes through [`LeaderCluster`].
//! This engine does not implement an ES remote transport client. Configured
//! addresses (including local/member addresses) therefore remain disconnected.
//! Never substitute this node's indices for a remote cluster: that fabricates
//! successful CCS/CCR results without a remote handshake or authorization.

use crate::error::{EsError, Result};
use crate::index::IndexService;
use crate::node::Node;
use serde_json::Value;
use std::sync::Arc;

/// The transport address this node publishes (`nodes.info`
/// `transport_address`).
pub const TRANSPORT_ADDRESS: &str = "127.0.0.1:9300";

/// `cluster.remote.<alias>.<key>` (transient, persistent, then node settings).
fn remote_setting(node: &Node, alias: &str, key: &str) -> Option<Value> {
    let full = format!("cluster.remote.{alias}.{key}");
    let nested = |m: &serde_json::Map<String, Value>| -> Option<Value> {
        if let Some(v) = m.get(&full) {
            return Some(v.clone());
        }
        let mut cur = m.get("cluster")?;
        for part in ["remote", alias, key] {
            cur = cur.get(part)?;
        }
        Some(cur.clone())
    };
    let v = {
        let cs = node.cluster_settings.read();
        nested(&cs.1).or_else(|| nested(&cs.0))
    };
    let v = v.or_else(|| nested(&node.node_settings.read()));
    v.filter(|v| !v.is_null())
}

fn str_list(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect(),
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

/// The configured addresses of `alias` (seeds, or the proxy address in proxy mode).
pub fn remote_addresses(node: &Node, alias: &str) -> Vec<String> {
    let mode = remote_setting(node, alias, "mode").and_then(|v| v.as_str().map(str::to_ascii_lowercase)).unwrap_or_else(|| "sniff".into());
    if mode == "proxy" {
        remote_setting(node, alias, "proxy_address").map(|v| str_list(&v)).unwrap_or_default()
    } else {
        remote_setting(node, alias, "seeds").map(|v| str_list(&v)).unwrap_or_default()
    }
}

/// Every configured remote cluster alias.
pub fn remote_aliases(node: &Node) -> Vec<String> {
    let mut out = std::collections::BTreeSet::new();
    let mut scan = |m: &serde_json::Map<String, Value>| {
        for k in m.keys() {
            if let Some(rest) = k.strip_prefix("cluster.remote.") {
                if let Some((alias, _)) = rest.split_once('.') {
                    out.insert(alias.to_string());
                }
            }
        }
        if let Some(r) = m.get("cluster").and_then(|c| c.get("remote")).and_then(Value::as_object) {
            out.extend(r.keys().cloned());
        }
    };
    {
        let cs = node.cluster_settings.read();
        scan(&cs.0);
        scan(&cs.1);
    }
    scan(&node.node_settings.read());
    out.into_iter().filter(|a| !remote_addresses(node, a).is_empty()).collect()
}

/// `NoSuchRemoteClusterException`.
pub fn no_such_remote(alias: &str) -> EsError {
    EsError::new(404, "no_such_remote_cluster_exception", format!("no such remote cluster: [{alias}]"))
}

/// The failure ES reports when a remote cluster cannot be reached (this engine
/// has no remote transport client, so every configured remote is unreachable):
/// sniff mode fails on the last seed with `ConnectTransportException`
/// (`status()` is BAD_GATEWAY, 502; message `[node name][address] msg`, a seed
/// node's name is empty); proxy mode with `NoSeedNodeLeftException` (500).
pub fn connection_error(node: &Node, alias: &str) -> EsError {
    let addresses = remote_addresses(node, alias);
    let Some(address) = addresses.last() else {
        return no_such_remote(alias);
    };
    tracing::debug!(alias, "remote cluster transport is not implemented: the remote cluster is unreachable");
    let mode = remote_setting(node, alias, "mode").and_then(|v| v.as_str().map(str::to_ascii_lowercase)).unwrap_or_else(|| "sniff".into());
    if mode == "proxy" {
        let server_name = remote_setting(node, alias, "server_name").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_else(|| "null".into());
        return EsError::new(500, "no_seed_node_left_exception", format!(
            "Unable to open any proxy connections to cluster [{alias}] at address [{address}], configuredAddress=[{address}], configuredServerName=[{server_name}]"
        ));
    }
    EsError::new(502, "connect_transport_exception", format!("[][{address}] connect_exception"))
}

/// `cluster:index` (not date math `<...>`, not a `::` selector).
fn split_remote(part: &str) -> Option<(&str, &str)> {
    if part.starts_with('<') { return None; }
    part.split_once(':').filter(|(alias, indices)| !alias.is_empty() && !indices.starts_with(':'))
}

/// Whether an index expression names any remote cluster (syntax only).
pub fn has_remote_expression(expr: &str) -> bool {
    expr.contains(':') && expr.split(',').any(|part| split_remote(part).is_some())
}

/// A remote cluster an expression names that was skipped (`skip_unavailable`).
#[derive(Clone, Debug)]
pub struct SkippedRemote {
    pub alias: String,
    /// The remote index expression (`SearchResponse.Cluster.indices`).
    pub indices: String,
    /// The connection failure that skipped it (`type`, `reason`).
    pub error_type: String,
    pub reason: String,
}

/// `RemoteClusterAware.groupClusterIndices` plus the unavailable-remote
/// policy. Every remote is unreachable here: a required one fails the whole
/// request, an optional one is skipped. No local index stands in for it.
#[derive(Clone, Debug, Default)]
pub struct CcsPlan {
    /// The local parts of the expression, in order.
    pub local: Vec<String>,
    pub skipped: Vec<SkippedRemote>,
}

impl CcsPlan {
    /// The local expression; `None` when the request names no local index
    /// (an empty expression would otherwise mean every local index).
    pub fn local_expr(&self) -> Option<String> {
        (!self.local.is_empty()).then(|| self.local.join(","))
    }

    pub fn aliases(&self) -> Vec<String> {
        self.skipped.iter().map(|s| s.alias.clone()).collect()
    }

    /// `SearchResponse.Clusters` (`_clusters`, with per-cluster `details`)
    /// for a search whose local part answered `local` (`None`: no local part).
    pub fn clusters(&self, local: Option<&Value>) -> Value {
        use serde_json::json;
        let mut details = serde_json::Map::new();
        let mut successful = 0;
        let mut partial = 0;
        let mut failed = 0;
        if let (Some(expr), Some(response)) = (self.local_expr(), local) {
            let shards = &response["_shards"];
            let (total, ok, bad) = (shards["total"].as_u64().unwrap_or(0), shards["successful"].as_u64().unwrap_or(0), shards["failed"].as_u64().unwrap_or(0));
            let timed_out = response["timed_out"].as_bool().unwrap_or(false);
            let status = if total > 0 && bad >= total { failed += 1; "failed" }
                else if timed_out || bad > 0 { partial += 1; "partial" }
                else { successful += 1; "successful" };
            let mut cluster = json!({"status": status, "indices": expr});
            if let Some(took) = response.get("took") { cluster["took"] = took.clone(); }
            cluster["timed_out"] = json!(timed_out);
            cluster["_shards"] = json!({"total": total, "successful": ok,
                "skipped": shards["skipped"].as_u64().unwrap_or(0), "failed": bad});
            if let Some(failures) = shards.get("failures").filter(|f| f.as_array().is_some_and(|a| !a.is_empty())) {
                cluster["failures"] = failures.clone();
            }
            details.insert("(local)".into(), cluster);
        }
        for skipped in &self.skipped {
            details.insert(skipped.alias.clone(), json!({
                "status": "skipped", "indices": skipped.indices, "timed_out": false,
                "failures": [{"shard": -1, "index": null, "reason": {"type": skipped.error_type, "reason": skipped.reason}}]
            }));
        }
        let total = details.len();
        json!({"total": total, "successful": successful, "skipped": self.skipped.len(), "running": 0,
            "partial": partial, "failed": failed, "details": details})
    }
}

/// `RemoteClusterAware.groupClusterIndices`: the local parts, in order, and
/// each matched remote alias with its index expressions. An unknown,
/// non-wildcard alias is `no_such_remote_cluster_exception`; a wildcard alias
/// that matches nothing is dropped. `-alias:*` drops clusters included
/// earlier in the list (an error for one that was not); `-alias:idx`
/// includes the cluster with `idx` excluded; `-alias:-idx` is rejected.
pub fn group_indices(node: &Node, expr: &str) -> Result<(Vec<String>, std::collections::BTreeMap<String, Vec<String>>)> {
    let configured = remote_aliases(node);
    let mut local = Vec::new();
    let mut remote_indices: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    let mut ever_included: std::collections::BTreeSet<String> = Default::default();
    let mut has_exclusions = false;
    for part in expr.split(',') {
        let Some((cluster, indices)) = split_remote(part) else {
            local.push(part.to_string());
            continue;
        };
        let exclude = cluster.starts_with('-');
        let pattern = if exclude { &cluster[1..] } else { cluster };
        let aliases: Vec<_> = if configured.iter().any(|a| a == pattern) { vec![pattern.to_string()] }
            else { configured.iter().filter(|a| crate::mapping::wildcard_match(pattern, a)).cloned().collect() };
        if aliases.is_empty() && !pattern.contains(['*', '?']) {
            return Err(no_such_remote(pattern));
        }
        if !exclude {
            for alias in aliases {
                ever_included.insert(alias.clone());
                remote_indices.entry(alias).or_default().push(indices.to_string());
            }
            continue;
        }
        has_exclusions = true;
        // `IndexNameExpressionResolver.splitSelectorExpression`.
        let (index_part, selector) = match indices.split_once("::") {
            Some((index, selector)) => (index, Some(selector)),
            None => (indices, None),
        };
        if index_part == "*" {
            if let Some(selector) = selector {
                return Err(EsError::illegal_argument(format!("To exclude a cluster you must not specify a selector, but found selector: [{selector}]")));
            }
            let mut failed = Vec::new();
            for alias in aliases {
                remote_indices.remove(&alias);
                if !ever_included.contains(&alias) { failed.push(alias); }
            }
            if !failed.is_empty() {
                let one = failed.len() == 1;
                return Err(EsError::illegal_argument(format!(
                    "Attempt to exclude cluster{} [{}] failed as {} not included in the list of clusters to be included (note: the \"include\" expression must precede the \"exclude\" expression)",
                    if one { "" } else { "s" }, failed.join(", "), if one { "it is" } else { "they are" })));
            }
        } else if index_part.starts_with('-') {
            return Err(EsError::illegal_argument(format!("cannot apply exclusion for both the cluster and the index expression, but found: [{cluster}:{index_part}]")));
        } else {
            // `-remote:logs` is sugar for including the cluster without `logs`.
            for alias in aliases {
                ever_included.insert(alias.clone());
                remote_indices.entry(alias).or_default().push(format!("-{indices}"));
            }
        }
    }
    if has_exclusions && local.is_empty() && remote_indices.is_empty() {
        return Err(EsError::illegal_argument(format!("The '-' exclusions in the index expression list excludes all indexes. Nothing to search. Input: [{expr}]")));
    }
    Ok((local, remote_indices))
}

/// `skip_unavailable` of a configured alias (true by default in 9.x).
pub fn skip_unavailable(node: &Node, alias: &str) -> bool {
    remote_info(node)[alias]["skip_unavailable"].as_bool().unwrap_or(true)
}

/// Plan an index expression's remote parts. Errors: an unknown, non-wildcard
/// alias (`no_such_remote_cluster_exception`) or any required remote (the
/// connection failure ES reports for it).
pub fn ccs_plan(node: &Node, expr: &str) -> Result<CcsPlan> {
    let info = remote_info(node);
    let (local, remote_indices) = group_indices(node, expr)?;
    let mut plan = CcsPlan { local, skipped: Vec::new() };
    for (alias, indices) in remote_indices {
        let error = connection_error(node, &alias);
        if !info[&alias]["skip_unavailable"].as_bool().unwrap_or(true) {
            return Err(error);
        }
        plan.skipped.push(SkippedRemote { alias, indices: indices.join(","), error_type: error.kind, reason: error.reason });
    }
    Ok(plan)
}

/// `_resolve/cluster` entries for the remote clusters an expression names
/// (all configured ones for the index-less form): none is reachable here, so
/// each is `{"connected": false, "skip_unavailable": ...}` as ES reports a
/// remote-unavailable failure. Returns the local expression, if any.
pub fn resolve_cluster_remotes(node: &Node, expr: Option<&str>) -> Result<(Option<String>, serde_json::Map<String, Value>)> {
    let (local, aliases): (Vec<String>, Vec<String>) = match expr {
        Some(expr) => { let (local, remotes) = group_indices(node, expr)?; (local, remotes.into_keys().collect()) }
        None => (Vec::new(), remote_aliases(node)),
    };
    let entries = aliases.into_iter().map(|alias| {
        let skip = skip_unavailable(node, &alias);
        (alias, serde_json::json!({"connected": false, "skip_unavailable": skip}))
    }).collect();
    Ok(((!local.is_empty()).then(|| local.join(",")), entries))
}

/// [`Node::resolve`] for APIs that accept cross-cluster expressions: the
/// remote parts are planned (required remotes fail, optional ones are
/// skipped) and only the local part resolves; a purely remote expression
/// resolves to nothing rather than to every local index.
pub fn resolve_local_part(node: &Node, expr: &str, opts: crate::node::ResolveOpts) -> Result<(Vec<crate::search::Target>, Option<CcsPlan>)> {
    if !has_remote_expression(expr) {
        return Ok((node.resolve(expr, opts)?, None));
    }
    let plan = ccs_plan(node, expr)?;
    let targets = match plan.local_expr() {
        Some(local) => node.resolve(&local, opts)?,
        None => Vec::new(),
    };
    Ok((targets, Some(plan)))
}

/// A connected leader cluster.
#[derive(Clone)]
pub enum LeaderCluster {
    /// Legacy local adapter retained for internal follow-task types. `connect`
    /// never constructs it; it is not a remote transport connection.
    Local(Arc<Node>),
}

impl LeaderCluster {
    /// `RemoteClusterService.getRemoteClusterClient(alias)` + connect.
    pub fn connect(node: &Arc<Node>, alias: &str) -> Result<LeaderCluster> {
        Err(connection_error(node, alias))
    }

    /// The leader's cluster name.
    pub fn cluster_name(&self) -> String {
        match self {
            LeaderCluster::Local(n) => n.cluster_name.clone(),
        }
    }

    /// A leader index by name (`None` when no concrete index has that name).
    pub fn index(&self, name: &str) -> Option<Arc<IndexService>> {
        match self {
            LeaderCluster::Local(n) => n.get_index(name),
        }
    }

    /// Every concrete leader index.
    pub fn indices(&self) -> Vec<Arc<IndexService>> {
        match self {
            LeaderCluster::Local(n) => n.all_indices(),
        }
    }

    /// What a non-index name refers to on the leader (`IndexAbstraction.Type`: `ALIAS` / `DATA_STREAM`).
    pub fn abstraction_type(&self, name: &str) -> Option<&'static str> {
        match self {
            LeaderCluster::Local(n) => {
                if n.data_streams.read().contains_key(name) {
                    return Some("DATA_STREAM");
                }
                if n.all_indices().iter().any(|s| s.aliases().contains_key(name)) {
                    return Some("ALIAS");
                }
                None
            }
        }
    }

    /// The data stream a leader index backs, if any.
    pub fn parent_data_stream(&self, index: &str) -> Option<String> {
        match self {
            LeaderCluster::Local(n) => n.data_streams.read().iter().find(|(_, d)| d.indices.iter().any(|i| i == index)).map(|(k, _)| k.clone()),
        }
    }

    /// The data stream a leader index backs, if any.
    pub fn data_stream_of(&self, index: &str) -> Option<crate::datastream::DataStream> {
        match self {
            LeaderCluster::Local(n) => n.data_streams.read().values().find(|d| d.indices.iter().any(|i| i == index)).cloned(),
        }
    }

    /// Whether the leader holds a CCR licence (platinum or better, active).
    pub fn check_license(&self) -> Result<()> {
        match self {
            LeaderCluster::Local(n) => {
                let lic = crate::license::current(n);
                if lic.effective_mode(crate::date::now_millis()).allows(crate::license::Mode::Platinum) {
                    Ok(())
                } else {
                    Err(EsError::new(
                        400,
                        "status_exception",
                        format!(
                            "can not fetch remote cluster state as the remote cluster [{}] is not licensed for [ccr]; the license mode [{}] on cluster [{}] does not enable [ccr]",
                            n.cluster_name,
                            lic.mode().name().to_ascii_uppercase(),
                            n.cluster_name
                        ),
                    ))
                }
            }
        }
    }
}

/// `GET /_remote/info` (`RemoteConnectionInfo.toXContent` per alias).
pub fn remote_info(node: &Node) -> Value {
    let mut out = serde_json::Map::new();
    let skip_default = !crate::node::ES_VERSION.starts_with("8.");
    for alias in remote_aliases(node) {
        let mode = remote_setting(node, &alias, "mode").and_then(|v| v.as_str().map(str::to_ascii_lowercase)).unwrap_or_else(|| "sniff".into());
        let addrs = remote_addresses(node, &alias);
        let connected = false;
        let setting = |k: &str| remote_setting(node, &alias, k).map(|v| match v {
            Value::String(s) => s,
            other => other.to_string(),
        });
        let mut m = serde_json::Map::new();
        m.insert("connected".into(), Value::Bool(connected));
        m.insert("mode".into(), Value::String(mode.clone()));
        if mode == "proxy" {
            m.insert("proxy_address".into(), Value::String(addrs.first().cloned().unwrap_or_default()));
            m.insert("server_name".into(), Value::String(setting("server_name").unwrap_or_default()));
            m.insert("num_proxy_sockets_connected".into(), Value::from(if connected { 18 } else { 0 }));
            m.insert("max_proxy_socket_connections".into(), Value::from(setting("proxy_socket_connections").and_then(|s| s.parse::<u64>().ok()).unwrap_or(18)));
        } else {
            m.insert("seeds".into(), Value::Array(addrs.iter().cloned().map(Value::String).collect()));
            m.insert("num_nodes_connected".into(), Value::from(if connected { 1 } else { 0 }));
            m.insert("max_connections_per_cluster".into(), Value::from(setting("node_connections").and_then(|s| s.parse::<u64>().ok()).unwrap_or(3)));
        }
        m.insert("initial_connect_timeout".into(), Value::String("30s".into()));
        let skip = setting("skip_unavailable").map(|s| s == "true").unwrap_or(skip_default);
        m.insert("skip_unavailable".into(), Value::Bool(skip));
        out.insert(alias, Value::Object(m));
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn configured_self_member_and_remote_addresses_are_not_connections() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        *node.transport_address.write() = "127.0.0.1:9301".into();
        *node.cluster_transport_addresses.write() = vec!["127.0.0.1:9300".into(), "127.0.0.1:9301".into()];
        node.cluster_settings.write().0.insert("cluster".into(), json!({"remote": {
            "self": {"seeds": ["127.0.0.1:9301"]},
            "member": {"seeds": ["127.0.0.1:9300"]},
            "remote": {"seeds": ["127.0.0.1:9400"]},
            "proxy": {"mode": "proxy", "proxy_address": "127.0.0.1:9301"}
        }}));
        let info = remote_info(&node);
        for alias in ["self", "member", "remote", "proxy"] {
            assert_eq!(info[alias]["connected"], false);
            let error = match LeaderCluster::connect(&node, alias) {
                Err(error) => error,
                Ok(_) => panic!("fabricated remote connection for {alias}"),
            };
            if alias == "proxy" {
                assert_eq!(error.status, 500);
                assert_eq!(error.kind, "no_seed_node_left_exception");
                assert_eq!(error.reason, "Unable to open any proxy connections to cluster [proxy] at address [127.0.0.1:9301], configuredAddress=[127.0.0.1:9301], configuredServerName=[null]");
            } else {
                assert_eq!(error.status, 502);
                assert_eq!(error.kind, "connect_transport_exception");
                assert!(error.reason.starts_with("[][127.0.0.1:") && error.reason.ends_with("] connect_exception"), "{}", error.reason);
            }
        }
        assert_eq!(info["member"]["num_nodes_connected"], 0);
        assert_eq!(info["proxy"]["num_proxy_sockets_connected"], 0);
        assert_eq!(connection_error(&node, "missing").kind, "no_such_remote_cluster_exception");
    }

    #[test]
    fn ccs_plan_skips_optional_fails_required_and_never_resolves_remote_locally() {
        let tmp = tempfile::tempdir().unwrap();
        let node = Node::open(tmp.path()).unwrap();
        node.create_index("logs", &json!({})).unwrap();
        node.cluster_settings.write().0.insert("cluster".into(), json!({"remote": {
            "cluster_one": {"seeds": ["127.0.0.1:9400"], "skip_unavailable": true},
            "required": {"seeds": ["127.0.0.1:9401"], "skip_unavailable": false}
        }}));
        assert!(has_remote_expression("logs,cluster_one:logs*"));
        assert!(!has_remote_expression("logs::failures,<logs-{now/d}>"));
        let plan = ccs_plan(&node, "logs,cluster_one:logs*").unwrap();
        assert_eq!(plan.local_expr().as_deref(), Some("logs"));
        assert_eq!(plan.skipped[0].alias, "cluster_one");
        assert_eq!(plan.skipped[0].indices, "logs*");
        let clusters = plan.clusters(Some(&json!({"took": 3, "timed_out": false, "_shards": {"total": 2, "successful": 2, "skipped": 0, "failed": 0}})));
        assert_eq!(clusters["total"], 2);
        assert_eq!(clusters["successful"], 1);
        assert_eq!(clusters["skipped"], 1);
        assert_eq!(clusters["details"]["(local)"]["status"], "successful");
        assert_eq!(clusters["details"]["(local)"]["_shards"]["total"], 2);
        assert_eq!(clusters["details"]["cluster_one"]["status"], "skipped");
        assert_eq!(clusters["details"]["cluster_one"]["failures"][0]["reason"]["type"], "connect_transport_exception");
        // Wildcard aliases that match nothing are dropped, as in ES.
        assert!(ccs_plan(&node, "logs,nomatch*:logs").unwrap().skipped.is_empty());
        assert_eq!(ccs_plan(&node, "logs,required:logs").unwrap_err().status, 502);
        assert_eq!(ccs_plan(&node, "*:logs").unwrap_err().status, 502, "a wildcard alias covering a required remote fails");
        assert_eq!(ccs_plan(&node, "missing:logs").unwrap_err().kind, "no_such_remote_cluster_exception");
        // `-cluster:*` drops a cluster included earlier (ES 9.5 groupClusterIndices).
        let plan = ccs_plan(&node, "logs,cluster_one:logs*,-cluster_one:*").unwrap();
        assert!(plan.skipped.is_empty());
        assert_eq!(plan.local_expr().as_deref(), Some("logs"));
        let error = ccs_plan(&node, "logs,-cluster_one:*").unwrap_err();
        assert_eq!(error.reason, "Attempt to exclude cluster [cluster_one] failed as it is not included in the list of clusters to be included (note: the \"include\" expression must precede the \"exclude\" expression)");
        assert!(ccs_plan(&node, "cluster_one:logs,-cluster_one:*::failures").unwrap_err().reason.contains("must not specify a selector, but found selector: [failures]"));
        assert_eq!(ccs_plan(&node, "cluster_one:logs,-cluster_one:-x").unwrap_err().reason, "cannot apply exclusion for both the cluster and the index expression, but found: [-cluster_one:-x]");
        assert!(ccs_plan(&node, "cluster_one:logs,-cluster_one:*").unwrap_err().reason.starts_with("The '-' exclusions in the index expression list excludes all indexes."));
        // `-cluster:idx` includes the cluster with `idx` excluded.
        let plan = ccs_plan(&node, "logs,-cluster_one:old").unwrap();
        assert_eq!(plan.skipped[0].indices, "-old");
        // Removals apply in list order, as ES's RemoteClusterAwareTests pin
        // (testGroupClusterIndicesExclusionThenInclusion, ...ExcludeThenReinclude):
        // an exclusion before any inclusion fails, and a later inclusion is kept.
        assert!(ccs_plan(&node, "-cluster_one:*,cluster_one:logs").unwrap_err().reason.starts_with("Attempt to exclude cluster [cluster_one] failed"));
        let plan = ccs_plan(&node, "cluster_one:a,-cluster_one:*,cluster_one:b").unwrap();
        assert_eq!(plan.skipped[0].indices, "b");
        // `-cluster:idx` before `-cluster:*` is excluded too: nothing left.
        assert!(ccs_plan(&node, "-cluster_one:logs,-cluster_one:*").unwrap_err().reason.starts_with("The '-' exclusions"));
        // A purely remote expression must not become "every local index".
        let (targets, plan) = resolve_local_part(&node, "cluster_one:logs", Default::default()).unwrap();
        assert!(targets.is_empty());
        assert_eq!(plan.unwrap().skipped.len(), 1);
        let (targets, _) = resolve_local_part(&node, "logs,cluster_one:logs", Default::default()).unwrap();
        assert_eq!(targets.len(), 1);
    }
}
