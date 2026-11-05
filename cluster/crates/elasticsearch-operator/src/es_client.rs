//! The operator's view of the Elasticsearch REST API, behind a trait so the
//! decision logic can be driven by an in-memory fake in tests.
//!
//! Endpoints used (all must behave like Elasticsearch):
//!
//! | call | endpoint |
//! |---|---|
//! | [`EsClient::health`] | `GET /_cluster/health` |
//! | [`EsClient::cluster_uuid`] | `GET /` (`cluster_uuid`, `"_na_"` before formation) |
//! | [`EsClient::nodes`] | `GET /_cat/nodes?format=json&full_id=true&h=id,name,node.role,master` |
//! | [`EsClient::shards`] | `GET /_cat/shards?format=json&h=index,shard,prirep,state,node` |
//! | [`EsClient::allocation_exclude_names`] | `GET /_cluster/settings?flat_settings=true` |
//! | [`EsClient::set_allocation_exclude_names`] | `PUT /_cluster/settings` (persistent `cluster.routing.allocation.exclude._name`) |
//! | [`EsClient::voting_exclusions`] | `GET /_cluster/state/metadata?filter_path=metadata.cluster_coordination.voting_config_exclusions` |
//! | [`EsClient::add_voting_exclusions`] | `POST /_cluster/voting_config_exclusions?node_names=a,b` |
//! | [`EsClient::clear_voting_exclusions`] | `DELETE /_cluster/voting_config_exclusions?wait_for_removal=false` |
//! | [`EsClient::shutdowns`] | `GET /_nodes/shutdown` |
//! | [`EsClient::put_shutdown`] | `PUT /_nodes/{id}/shutdown` |
//! | [`EsClient::delete_shutdown`] | `DELETE /_nodes/{id}/shutdown` |

use async_trait::async_trait;
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum EsError {
    #[error("elasticsearch unreachable: {0}")]
    Unreachable(String),
    #[error("elasticsearch returned {status}: {body}")]
    Status { status: u16, body: String },
    #[error("unexpected response: {0}")]
    Decode(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Health {
    Red,
    Yellow,
    Green,
}

impl Health {
    pub fn parse(s: &str) -> Option<Health> {
        match s.to_ascii_lowercase().as_str() {
            "green" => Some(Health::Green),
            "yellow" => Some(Health::Yellow),
            "red" => Some(Health::Red),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Health::Green => "green",
            Health::Yellow => "yellow",
            Health::Red => "red",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClusterHealth {
    pub status: Health,
    pub number_of_nodes: i64,
    pub relocating_shards: i64,
    pub initializing_shards: i64,
    pub unassigned_shards: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EsNode {
    pub id: String,
    pub name: String,
    /// `_cat/nodes` `node.role` letters (`m` = master-eligible, `d` = data, ...).
    pub roles: String,
    pub is_master: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardRow {
    pub index: String,
    pub shard: String,
    pub primary: bool,
    pub state: String,
    /// Node name; `None` when unassigned. For a relocating shard `_cat/shards`
    /// prints `src -> ip id dst`; we keep the source name.
    pub node: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShutdownType {
    Restart,
    Remove,
}

impl ShutdownType {
    pub fn as_str(self) -> &'static str {
        match self {
            ShutdownType::Restart => "restart",
            ShutdownType::Remove => "remove",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShutdownRecord {
    pub node_id: String,
    pub type_: ShutdownType,
    pub reason: String,
    /// `NOT_STARTED`, `IN_PROGRESS`, `STALLED` or `COMPLETE`.
    pub status: String,
}

#[async_trait]
pub trait EsClient: Send + Sync {
    async fn health(&self) -> Result<ClusterHealth, EsError>;
    /// `Some(uuid)` once the cluster has formed.
    async fn cluster_uuid(&self) -> Result<Option<String>, EsError>;
    async fn nodes(&self) -> Result<Vec<EsNode>, EsError>;
    async fn shards(&self) -> Result<Vec<ShardRow>, EsError>;
    async fn allocation_exclude_names(&self) -> Result<Vec<String>, EsError>;
    /// Empty slice clears the setting.
    async fn set_allocation_exclude_names(&self, names: &[String]) -> Result<(), EsError>;
    async fn voting_exclusions(&self) -> Result<Vec<String>, EsError>;
    async fn add_voting_exclusions(&self, names: &[String]) -> Result<(), EsError>;
    async fn clear_voting_exclusions(&self) -> Result<(), EsError>;
    async fn shutdowns(&self) -> Result<Vec<ShutdownRecord>, EsError>;
    async fn put_shutdown(&self, node_id: &str, type_: ShutdownType, reason: &str) -> Result<(), EsError>;
    async fn delete_shutdown(&self, node_id: &str) -> Result<(), EsError>;
}

/// Everything the planners need from Elasticsearch, fetched in one go.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EsObservation {
    pub health: Option<ClusterHealth>,
    pub nodes: Vec<EsNode>,
    pub shards: Vec<ShardRow>,
    pub allocation_excludes: Vec<String>,
    pub voting_exclusions: Vec<String>,
    pub shutdowns: Vec<ShutdownRecord>,
}

impl EsObservation {
    pub fn reachable(&self) -> bool {
        self.health.is_some()
    }
    pub fn node_id(&self, name: &str) -> Option<&str> {
        self.nodes.iter().find(|n| n.name == name).map(|n| n.id.as_str())
    }
}

/// Fetch an observation. An unreachable cluster yields `health: None` and
/// empty collections rather than an error: the planners treat that case
/// explicitly.
pub async fn observe(es: &dyn EsClient) -> Result<EsObservation, EsError> {
    let health = match es.health().await {
        Ok(h) => h,
        Err(EsError::Unreachable(_)) => return Ok(EsObservation::default()),
        // 503/408 from health = no master / timed out: unknown health.
        Err(EsError::Status { status, .. }) if status == 503 || status == 408 => return Ok(EsObservation::default()),
        Err(e) => return Err(e),
    };
    Ok(EsObservation {
        health: Some(health),
        nodes: es.nodes().await?,
        shards: es.shards().await?,
        allocation_excludes: es.allocation_exclude_names().await?,
        voting_exclusions: es.voting_exclusions().await?,
        shutdowns: es.shutdowns().await?,
    })
}

// ---------------------------------------------------------------------------
// HTTP implementation

pub struct HttpEsClient {
    base: String,
    http: reqwest::Client,
    auth: Option<(String, String)>,
}

impl HttpEsClient {
    pub fn new(base: impl Into<String>, timeout: Duration, auth: Option<(String, String)>) -> Result<Self, EsError> {
        let http = reqwest::Client::builder().timeout(timeout).build().map_err(|e| EsError::Unreachable(e.to_string()))?;
        Ok(HttpEsClient { base: base.into().trim_end_matches('/').to_string(), http, auth })
    }

    async fn call(&self, method: reqwest::Method, path: &str, body: Option<Value>) -> Result<Value, EsError> {
        let mut req = self.http.request(method, format!("{}{path}", self.base));
        if let Some((u, p)) = &self.auth {
            req = req.basic_auth(u, Some(p));
        }
        if let Some(b) = body {
            req = req.json(&b);
        }
        let resp = req.send().await.map_err(|e| EsError::Unreachable(e.to_string()))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.map_err(|e| EsError::Unreachable(e.to_string()))?;
        if !(200..300).contains(&status) {
            return Err(EsError::Status { status, body: text });
        }
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|e| EsError::Decode(format!("{path}: {e}")))
    }

    async fn get(&self, path: &str) -> Result<Value, EsError> {
        self.call(reqwest::Method::GET, path, None).await
    }
}

fn s(v: &Value, k: &str) -> String {
    match v.get(k) {
        Some(Value::String(x)) => x.clone(),
        Some(Value::Null) | None => String::new(),
        Some(o) => o.to_string(),
    }
}

fn i(v: &Value, k: &str) -> i64 {
    v.get(k).and_then(|x| x.as_i64().or_else(|| x.as_str().and_then(|s| s.parse().ok()))).unwrap_or(0)
}

pub fn parse_health(v: &Value) -> Result<ClusterHealth, EsError> {
    let status = Health::parse(&s(v, "status")).ok_or_else(|| EsError::Decode(format!("health status in {v}")))?;
    Ok(ClusterHealth {
        status,
        number_of_nodes: i(v, "number_of_nodes"),
        relocating_shards: i(v, "relocating_shards"),
        initializing_shards: i(v, "initializing_shards"),
        unassigned_shards: i(v, "unassigned_shards"),
    })
}

pub fn parse_cat_nodes(v: &Value) -> Vec<EsNode> {
    v.as_array()
        .map(|a| a.iter().map(|n| EsNode { id: s(n, "id"), name: s(n, "name"), roles: s(n, "node.role"), is_master: s(n, "master") == "*" }).collect())
        .unwrap_or_default()
}

pub fn parse_cat_shards(v: &Value) -> Vec<ShardRow> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|r| {
                    let node = s(r, "node");
                    let node = node.split(" -> ").next().unwrap_or("").trim().to_string();
                    ShardRow {
                        index: s(r, "index"),
                        shard: s(r, "shard"),
                        primary: s(r, "prirep") == "p",
                        state: s(r, "state"),
                        node: if node.is_empty() { None } else { Some(node) },
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

pub const ALLOCATION_EXCLUDE_NAME: &str = "cluster.routing.allocation.exclude._name";

pub fn parse_exclude_names(v: &Value) -> Vec<String> {
    // transient wins over persistent, like ES.
    for scope in ["transient", "persistent"] {
        if let Some(x) = v.get(scope).and_then(|p| p.get(ALLOCATION_EXCLUDE_NAME)) {
            let raw = match x {
                Value::String(s) => s.clone(),
                Value::Array(a) => a.iter().filter_map(|e| e.as_str()).collect::<Vec<_>>().join(","),
                _ => String::new(),
            };
            return raw.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect();
        }
    }
    Vec::new()
}

pub fn parse_voting_exclusions(v: &Value) -> Vec<String> {
    v.pointer("/metadata/cluster_coordination/voting_config_exclusions")
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|e| s(e, "node_name")).filter(|n| !n.is_empty() && n != "_absent_").collect())
        .unwrap_or_default()
}

pub fn parse_shutdowns(v: &Value) -> Vec<ShutdownRecord> {
    v.get("nodes")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|n| ShutdownRecord {
                    node_id: s(n, "node_id"),
                    type_: if s(n, "type").eq_ignore_ascii_case("restart") { ShutdownType::Restart } else { ShutdownType::Remove },
                    reason: s(n, "reason"),
                    status: s(n, "status").to_ascii_uppercase(),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn enc(s: &str) -> String {
    s.bytes()
        .map(|b| if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~' | b',') { (b as char).to_string() } else { format!("%{b:02X}") })
        .collect()
}

#[async_trait]
impl EsClient for HttpEsClient {
    async fn health(&self) -> Result<ClusterHealth, EsError> {
        parse_health(&self.get("/_cluster/health").await?)
    }

    async fn cluster_uuid(&self) -> Result<Option<String>, EsError> {
        let v = self.get("/").await?;
        let u = s(&v, "cluster_uuid");
        Ok(if u.is_empty() || u == "_na_" { None } else { Some(u) })
    }

    async fn nodes(&self) -> Result<Vec<EsNode>, EsError> {
        Ok(parse_cat_nodes(&self.get("/_cat/nodes?format=json&full_id=true&h=id,name,node.role,master").await?))
    }

    async fn shards(&self) -> Result<Vec<ShardRow>, EsError> {
        Ok(parse_cat_shards(&self.get("/_cat/shards?format=json&h=index,shard,prirep,state,node").await?))
    }

    async fn allocation_exclude_names(&self) -> Result<Vec<String>, EsError> {
        Ok(parse_exclude_names(&self.get("/_cluster/settings?flat_settings=true").await?))
    }

    async fn set_allocation_exclude_names(&self, names: &[String]) -> Result<(), EsError> {
        let value = if names.is_empty() { Value::Null } else { Value::String(names.join(",")) };
        self.call(reqwest::Method::PUT, "/_cluster/settings", Some(json!({ "persistent": { ALLOCATION_EXCLUDE_NAME: value } }))).await?;
        Ok(())
    }

    async fn voting_exclusions(&self) -> Result<Vec<String>, EsError> {
        Ok(parse_voting_exclusions(&self.get("/_cluster/state/metadata?filter_path=metadata.cluster_coordination.voting_config_exclusions").await?))
    }

    async fn add_voting_exclusions(&self, names: &[String]) -> Result<(), EsError> {
        let path = format!("/_cluster/voting_config_exclusions?node_names={}", enc(&names.join(",")));
        self.call(reqwest::Method::POST, &path, None).await?;
        Ok(())
    }

    async fn clear_voting_exclusions(&self) -> Result<(), EsError> {
        self.call(reqwest::Method::DELETE, "/_cluster/voting_config_exclusions?wait_for_removal=false", None).await?;
        Ok(())
    }

    async fn shutdowns(&self) -> Result<Vec<ShutdownRecord>, EsError> {
        Ok(parse_shutdowns(&self.get("/_nodes/shutdown").await?))
    }

    async fn put_shutdown(&self, node_id: &str, type_: ShutdownType, reason: &str) -> Result<(), EsError> {
        let mut body = json!({ "type": type_.as_str(), "reason": reason });
        if type_ == ShutdownType::Restart {
            body["allocation_delay"] = json!("5m");
        }
        self.call(reqwest::Method::PUT, &format!("/_nodes/{}/shutdown", enc(node_id)), Some(body)).await?;
        Ok(())
    }

    async fn delete_shutdown(&self, node_id: &str) -> Result<(), EsError> {
        match self.call(reqwest::Method::DELETE, &format!("/_nodes/{}/shutdown", enc(node_id)), None).await {
            Err(EsError::Status { status: 404, .. }) => Ok(()),
            other => other.map(|_| ()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_es_responses() {
        let h = parse_health(&json!({"cluster_name":"c","status":"yellow","number_of_nodes":3,"relocating_shards":1,"initializing_shards":0,"unassigned_shards":2})).unwrap();
        assert_eq!(h.status, Health::Yellow);
        assert_eq!(h.unassigned_shards, 2);
        assert!(parse_health(&json!({"status":"purple"})).is_err());

        let n = parse_cat_nodes(&json!([{"id":"abc","name":"c-es-m-0","node.role":"m","master":"*"},{"id":"def","name":"c-es-d-0","node.role":"d","master":"-"}]));
        assert_eq!(n.len(), 2);
        assert!(n[0].is_master && !n[1].is_master);

        let sh = parse_cat_shards(&json!([
            {"index":"i","shard":"0","prirep":"p","state":"STARTED","node":"a"},
            {"index":"i","shard":"0","prirep":"r","state":"UNASSIGNED","node":null},
            {"index":"i","shard":"1","prirep":"p","state":"RELOCATING","node":"a -> 10.0.0.2 xyz b"}
        ]));
        assert_eq!(sh[0].node.as_deref(), Some("a"));
        assert_eq!(sh[1].node, None);
        assert_eq!(sh[2].node.as_deref(), Some("a"));

        assert_eq!(parse_exclude_names(&json!({"persistent":{ALLOCATION_EXCLUDE_NAME:"a, b"},"transient":{}})), vec!["a", "b"]);
        assert_eq!(parse_exclude_names(&json!({"persistent":{},"transient":{ALLOCATION_EXCLUDE_NAME:"t"}})), vec!["t"]);
        assert!(parse_exclude_names(&json!({"persistent":{}})).is_empty());

        assert_eq!(parse_voting_exclusions(&json!({"metadata":{"cluster_coordination":{"voting_config_exclusions":[{"node_id":"x","node_name":"m-2"}]}}})), vec!["m-2"]);
        assert!(parse_voting_exclusions(&json!({})).is_empty());

        let sd = parse_shutdowns(&json!({"nodes":[{"node_id":"x","type":"RESTART","reason":"r","status":"COMPLETE"}]}));
        assert_eq!(sd[0].type_, ShutdownType::Restart);
        assert_eq!(sd[0].status, "COMPLETE");
        assert_eq!(enc("a b,c"), "a%20b,c");
    }
}
