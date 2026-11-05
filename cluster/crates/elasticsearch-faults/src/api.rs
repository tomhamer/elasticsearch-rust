//! HTTP plumbing: requests with timeouts, outcome classification, and
//! control-plane queries (who is master / primary, wait for convergence).

use crate::cluster::{Cluster, ProcState};
use crate::history::Outcome;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone)]
pub enum HttpErr {
    /// The request never reached the server (connection refused / connect
    /// timeout): definitely not executed.
    NotSent(String),
    /// Sent (or possibly sent) but no response: indeterminate.
    Unknown(String),
}

impl HttpErr {
    pub fn msg(&self) -> String {
        match self {
            HttpErr::NotSent(m) => format!("not sent: {m}"),
            HttpErr::Unknown(m) => format!("no response: {m}"),
        }
    }
}

pub fn client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(Duration::from_millis(1500))
        .pool_idle_timeout(Duration::from_secs(5))
        .pool_max_idle_per_host(8)
        .build()
        .expect("http client")
}

pub async fn call(c: &reqwest::Client, method: &str, url: &str, body: Option<String>, timeout: Option<Duration>) -> Result<(u16, Value), HttpErr> {
    let m = reqwest::Method::from_bytes(method.as_bytes()).expect("method");
    let mut rb = c.request(m, url);
    if let Some(t) = timeout {
        rb = rb.timeout(t);
    }
    if let Some(b) = body {
        let ct = if url.contains("_bulk") { "application/x-ndjson" } else { "application/json" };
        rb = rb.header("content-type", ct).body(b);
    }
    let resp = rb.send().await.map_err(|e| if e.is_connect() { HttpErr::NotSent(e.to_string()) } else { HttpErr::Unknown(e.to_string()) })?;
    let status = resp.status().as_u16();
    let text = resp.text().await.map_err(|e| HttpErr::Unknown(e.to_string()))?;
    let v = serde_json::from_str(&text).unwrap_or(Value::String(text));
    Ok((status, v))
}

pub fn error_type(v: &Value) -> String {
    v["error"]["type"].as_str().or_else(|| v["error"].as_str()).unwrap_or("").to_string()
}

/// Classify a write response (single-doc or one bulk item).
pub fn classify_write(status: u16, body: &Value) -> Outcome {
    match status {
        200..=299 => Outcome::Ok,
        // Delete of a missing doc completed on the primary.
        404 if body["result"] == "not_found" => Outcome::Ok,
        409 | 400 => Outcome::Fail,
        // Blocks are checked before anything executes.
        503 if error_type(body) == "cluster_block_exception" => Outcome::Fail,
        _ => Outcome::Info,
    }
}

/// Thin wrapper for control-plane calls.
pub struct Api {
    pub http: reqwest::Client,
    pub cluster: Arc<Cluster>,
}

#[derive(Clone, Debug, Default)]
pub struct Routing {
    /// index -> (primary node, replica nodes) among STARTED copies.
    pub started: BTreeMap<String, (Option<usize>, Vec<usize>)>,
    /// index -> nodes with an INITIALIZING copy.
    pub initializing: BTreeMap<String, Vec<usize>>,
}

impl Api {
    pub fn new(cluster: Arc<Cluster>) -> Api {
        Api { http: client(Duration::from_secs(5)), cluster }
    }

    pub async fn get(&self, i: usize, path: &str, timeout: Duration) -> Result<(u16, Value), HttpErr> {
        call(&self.http, "GET", &format!("{}{}", self.cluster.url(i), path), None, Some(timeout)).await
    }

    pub async fn send(&self, i: usize, method: &str, path: &str, body: Option<String>, timeout: Duration) -> Result<(u16, Value), HttpErr> {
        call(&self.http, method, &format!("{}{}", self.cluster.url(i), path), body, Some(timeout)).await
    }

    /// The master as seen by a plurality of running nodes.
    pub async fn master(&self) -> Option<usize> {
        let mut votes: BTreeMap<usize, usize> = BTreeMap::new();
        let futs = self.cluster.running().into_iter().map(|i| async move { self.get(i, "/_cluster/state/master_node,nodes?filter_path=master_node,nodes.*.name", Duration::from_secs(2)).await });
        for (s, v) in futures::future::join_all(futs).await.into_iter().flatten() {
            if s != 200 {
                continue;
            }
            let Some(m) = v["master_node"].as_str() else { continue };
            if let Some(idx) = v["nodes"][m]["name"].as_str().and_then(Cluster::index_of_name) {
                *votes.entry(idx).or_default() += 1;
            }
        }
        votes.into_iter().max_by_key(|(_, c)| *c).map(|(i, _)| i)
    }

    /// Routing as seen by the master (or, failing that, any running node).
    pub async fn routing(&self) -> Option<Routing> {
        let mut order = Vec::new();
        if let Some(m) = self.master().await {
            order.push(m);
        }
        order.extend(self.cluster.running());
        for i in order {
            if self.cluster.state(i) != ProcState::Running {
                continue;
            }
            let Ok((200, v)) = self.get(i, "/_cluster/state/nodes,routing_table", Duration::from_secs(2)).await else { continue };
            return Some(parse_routing(&v));
        }
        None
    }

    pub async fn primary(&self, index: &str) -> Option<usize> {
        self.routing().await.and_then(|r| r.started.get(index).and_then(|(p, _)| *p))
    }

    /// Wait until every running node reports green, the same master, the
    /// same state version and all `expect` nodes. Returns elapsed seconds or
    /// a description of the last observed disagreement.
    pub async fn wait_converged(&self, expect: usize, timeout: Duration) -> Result<f64, String> {
        let start = tokio::time::Instant::now();
        loop {
            let running = self.cluster.running();
            let futs = running.iter().map(|&i| async move {
                (i, self.get(i, "/_cluster/health?local=true", Duration::from_secs(3)).await, self.get(i, "/_cluster/state/master_node,version", Duration::from_secs(3)).await)
            });
            let mut ok = running.len() == expect;
            let mut seen: BTreeSet<(String, i64)> = BTreeSet::new();
            let mut desc = Vec::new();
            for (i, h, s) in futures::future::join_all(futs).await {
                match (h, s) {
                    (Ok((_, h)), Ok((200, s))) => {
                        let status = h["status"].as_str().unwrap_or("?").to_string();
                        let nodes = h["number_of_nodes"].as_u64().unwrap_or(0) as usize;
                        let master = s["master_node"].as_str().unwrap_or("-").to_string();
                        let ver = s["version"].as_i64().unwrap_or(-1);
                        desc.push(format!("node-{}:{}/{}n/m={}/v{}", i + 1, status, nodes, &master[..master.len().min(6)], ver));
                        if status != "green" || nodes != expect || master == "-" {
                            ok = false;
                        }
                        seen.insert((master, ver));
                    }
                    (h, s) => {
                        ok = false;
                        desc.push(format!("node-{}:unreachable({:?}/{:?})", i + 1, h.err().map(|e| e.msg()), s.err().map(|e| e.msg())));
                    }
                }
            }
            if ok && seen.len() == 1 {
                return Ok(start.elapsed().as_secs_f64());
            }
            if start.elapsed() >= timeout {
                return Err(desc.join(" "));
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}

pub fn parse_routing(v: &Value) -> Routing {
    let name_of = |id: &str| v["nodes"][id]["name"].as_str().and_then(Cluster::index_of_name);
    let mut r = Routing::default();
    if let Some(indices) = v["routing_table"]["indices"].as_object() {
        for (name, ix) in indices {
            let mut prim = None;
            let mut reps = Vec::new();
            let mut init = Vec::new();
            for copy in ix["shards"]["0"].as_array().cloned().unwrap_or_default() {
                let Some(node) = copy["node"].as_str().and_then(name_of) else { continue };
                match copy["state"].as_str() {
                    Some("STARTED") | Some("RELOCATING") => {
                        if copy["primary"].as_bool() == Some(true) {
                            prim = Some(node);
                        } else {
                            reps.push(node);
                        }
                    }
                    Some("INITIALIZING") => init.push(node),
                    _ => {}
                }
            }
            r.started.insert(name.clone(), (prim, reps));
            r.initializing.insert(name.clone(), init);
        }
    }
    r
}
