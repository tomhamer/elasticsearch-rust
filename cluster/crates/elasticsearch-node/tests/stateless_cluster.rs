//! Stateless clusters as real `elasticsearch-node` processes (so nodes can
//! be killed): reads go to search copies only, shards without a search
//! copy fail like unassigned shards, and (object store) a lost index node
//! or search node is rebuilt from the object store.

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// One node process; killed (SIGKILL) when dropped.
struct Node {
    child: Option<Child>,
    http: u16,
    transport: u16,
    log: PathBuf,
}

impl Node {
    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.http)
    }

    fn kill(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }

    fn log_text(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        self.kill();
    }
}

struct Spec<'a> {
    cluster: &'a str,
    name: &'a str,
    roles: &'a str,
    /// Transport ports of the nodes started before it (its seed hosts).
    seeds: &'a [u16],
    masters: &'a [&'a str],
    extra: &'a [(&'a str, &'a str)],
}

/// Start a node on OS-assigned ports and wait until it reports the ports it
/// bound (`node.portsfile`); started again if a bind failed meanwhile.
async fn spawn(dir: &Path, s: Spec<'_>) -> Node {
    let bin = env!("CARGO_BIN_EXE_elasticsearch-node");
    let log = dir.join(format!("{}.log", s.name));
    let logs = dir.join(format!("{}-logs", s.name));
    for _ in 0..8 {
        ports::clear_ports_files(&logs);
        let offset = std::fs::metadata(&log).map(|m| m.len()).unwrap_or(0) as usize;
        let out = std::fs::OpenOptions::new().create(true).append(true).open(&log).unwrap();
        let mut args: Vec<String> = Vec::new();
        let mut e = |k: &str, v: &str| {
            args.push("-E".into());
            args.push(format!("{k}={v}"));
        };
        e("cluster.name", s.cluster);
        e("node.name", s.name);
        e("node.roles", s.roles);
        e("path.data", &dir.join(s.name).to_string_lossy());
        e("path.logs", &logs.to_string_lossy());
        e("path.repo", &dir.join("repo").to_string_lossy());
        e("network.host", "127.0.0.1");
        e("http.port", "0");
        e("transport.port", "0");
        e("node.portsfile", "true");
        let seeds = ports::seeds(s.seeds);
        if !seeds.is_empty() {
            e("discovery.seed_hosts", &seeds.join(","));
        }
        e("cluster.initial_master_nodes", &s.masters.join(","));
        e("xpack.security.enabled", "false");
        e("stateless.enabled", "true");
        e("stateless.object_store.type", "fs");
        e("stateless.object_store.bucket", "stateless");
        e("stateless.object_store.base_path", "base_path");
        e("stateless.translog.flush.interval", "20ms");
        for w in ["low", "high", "flood_stage"] {
            e(&format!("cluster.routing.allocation.disk.watermark.{w}"), "1b");
        }
        for (k, v) in s.extra {
            e(k, v);
        }
        let child = Command::new(bin).args(&args).stdin(Stdio::null()).stdout(out.try_clone().unwrap()).stderr(out).spawn().expect("spawn elasticsearch-node");
        // Killed on drop, also if a check below fails.
        let mut node = Node { child: Some(child), http: 0, transport: 0, log: log.clone() };
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let exited = node.child.as_mut().expect("spawned").try_wait().expect("poll child");
            if let Some(status) = exited {
                let text = std::fs::read(&log).unwrap_or_default();
                let since = String::from_utf8_lossy(&text[offset.min(text.len())..]).into_owned();
                assert!(ports::is_addr_in_use(&since), "{} exited ({status}) before binding:\n{since}", s.name);
                node.child = None;
                break;
            }
            // `http.ports` is written after `transport.ports`.
            if let Some(http) = ports::read_ports_file(&logs, "http") {
                if let Some(transport) = ports::read_ports_file(&logs, "transport") {
                    node.http = http;
                    node.transport = transport;
                    return node;
                }
            }
            assert!(Instant::now() < deadline, "{} did not report its ports: {}", s.name, node.log_text());
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    panic!("{} found no free ports: {}", s.name, std::fs::read_to_string(&log).unwrap_or_default());
}

async fn send(c: &reqwest::Client, method: &str, url: &str, body: Option<Value>) -> (u16, Value) {
    let m = reqwest::Method::from_bytes(method.as_bytes()).unwrap();
    let mut r = c.request(m, url);
    if let Some(b) = body {
        r = r.json(&b);
    }
    let resp = r.send().await.unwrap_or_else(|e| panic!("{method} {url}: {e}"));
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap();
    (status, serde_json::from_str(&text).unwrap_or(Value::String(text)))
}

async fn wait_for(what: &str, timeout: Duration, mut f: impl AsyncFnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if f().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("timed out waiting for {what}");
}

async fn wait_nodes(c: &reqwest::Client, n: &Node, count: usize) {
    wait_for(&format!("{count} nodes"), Duration::from_secs(90), async || {
        let r = c.get(n.url(&format!("/_cluster/health?wait_for_nodes={count}&timeout=1s"))).send().await;
        match r {
            Ok(r) => r.json::<Value>().await.is_ok_and(|v| v["number_of_nodes"] == count && v["timed_out"] == false),
            Err(_) => false,
        }
    })
    .await;
}

async fn wait_green(c: &reqwest::Client, n: &Node, index: &str) {
    wait_for(&format!("{index} green"), Duration::from_secs(90), async || {
        let (st, v) = send(c, "GET", &n.url(&format!("/_cluster/health/{index}?wait_for_status=green&timeout=1s")), None).await;
        st == 200 && v["status"] == "green"
    })
    .await;
}

/// `query_total` / get `total` this node's copies served.
async fn served(c: &reqwest::Client, n: &Node) -> (u64, u64) {
    let (_, v) = send(c, "GET", &n.url("/_nodes/_local/stats/indices/search,get"), None).await;
    let node = v["nodes"].as_object().and_then(|m| m.values().next()).cloned().unwrap_or(Value::Null);
    (node["indices"]["search"]["query_total"].as_u64().unwrap_or(0), node["indices"]["get"]["total"].as_u64().unwrap_or(0))
}

const CLUSTER_A: &str = "stateless-routing";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reads_go_to_search_copies_and_missing_search_copies_fail_like_es() {
    let dir = evidence::retained_dir("stateless_cluster");
    let index_node = spawn(dir.path(), Spec { cluster: CLUSTER_A, name: "index", roles: "master,remote_cluster_client,ingest,index", seeds: &[], masters: &["index"], extra: &[] }).await;
    let mut search_node = spawn(dir.path(), Spec { cluster: CLUSTER_A, name: "search", roles: "remote_cluster_client,search", seeds: &[index_node.transport], masters: &["index"], extra: &[] }).await;
    let c = reqwest::Client::new();
    wait_nodes(&c, &index_node, 2).await;
    let i = &index_node;
    // `wait_for_active_shards` 1 on a stateless index means a search shard
    // per shard (ES `ActiveShardCount.enoughShardsActive`), so a realtime
    // get right after creating the index finds a search copy. One recovery
    // at a time keeps most search copies initializing when the primaries
    // are already active.
    let (st, _) = send(&c, "PUT", &i.url("/_cluster/settings"), Some(json!({"persistent": {"cluster.routing.allocation.node_concurrent_recoveries": 1}}))).await;
    assert_eq!(st, 200);
    let (st, v) = send(&c, "PUT", &i.url("/fresh?wait_for_active_shards=1"), Some(json!({"settings": {"number_of_shards": 8, "number_of_replicas": 1}}))).await;
    assert_eq!(st, 200, "{v}");
    let (_, shards) = send(&c, "GET", &i.url("/_cat/shards/fresh?format=json&h=shard,prirep,state"), None).await;
    let started = shards.as_array().unwrap().iter().filter(|s| s["prirep"] == "r" && s["state"] == "STARTED").count();
    assert_eq!(started, 8, "every shard has an active search shard once the index is created: {shards}");
    for id in 0..8 {
        let (st, _) = send(&c, "PUT", &i.url(&format!("/fresh/_doc/{id}")), Some(json!({"t": "x"}))).await;
        assert_eq!(st, 201);
        let (st, v) = send(&c, "GET", &i.url(&format!("/fresh/_doc/{id}")), None).await;
        assert_eq!((st, v["found"].clone()), (200, json!(true)), "{v}");
    }
    let (st, _) = send(&c, "PUT", &i.url("/_cluster/settings"), Some(json!({"persistent": {"cluster.routing.allocation.node_concurrent_recoveries": null}}))).await;
    assert_eq!(st, 200);
    for (name, shards, replicas) in [("s", 1, 1), ("m", 3, 1), ("solo", 1, 0)] {
        let (st, v) = send(&c, "PUT", &i.url(&format!("/{name}")), Some(json!({"settings": {"number_of_shards": shards, "number_of_replicas": replicas}}))).await;
        assert_eq!(st, 200, "{v}");
    }
    wait_green(&c, i, "s,m,solo").await;
    for id in 0..9 {
        for name in ["s", "m", "solo"] {
            let (st, v) = send(&c, "PUT", &i.url(&format!("/{name}/_doc/{id}")), Some(json!({"t": "hello world", "n": id}))).await;
            assert!(st == 200 || st == 201, "{v}");
        }
    }
    let (st, _) = send(&c, "POST", &i.url("/s,m,solo/_refresh"), None).await;
    assert_eq!(st, 200);
    let before = (served(&c, i).await, served(&c, &search_node).await);
    // Every read, coordinated by either node, runs on the search copies.
    for coord in [i, &search_node] {
        let (st, v) = send(&c, "GET", &coord.url("/s,m/_search?size=0"), None).await;
        assert_eq!((st, v["hits"]["total"]["value"].clone()), (200, json!(18)), "{v}");
        let (_, v) = send(&c, "GET", &coord.url("/m/_count"), None).await;
        assert_eq!(v["count"], 9, "a refresh reaches every shard copy: {v}");
        let (_, v) = send(&c, "GET", &coord.url("/s/_doc/3"), None).await;
        assert_eq!(v["found"], true, "{v}");
        let (_, v) = send(&c, "GET", &coord.url("/m/_doc/4"), None).await;
        assert_eq!(v["found"], true, "{v}");
        let (_, v) = send(&c, "POST", &coord.url("/_mget"), Some(json!({"docs": [{"_index": "s", "_id": "1"}, {"_index": "m", "_id": "2"}]}))).await;
        assert!(v["docs"].as_array().unwrap().iter().all(|d| d["found"] == true), "{v}");
        let resp = c.post(coord.url("/_msearch")).header("content-type", "application/x-ndjson").body("{\"index\":\"s\"}\n{\"size\":0}\n{\"index\":\"m\"}\n{\"size\":0}\n").send().await.unwrap();
        let v: Value = resp.json().await.unwrap();
        assert_eq!(v["responses"][1]["hits"]["total"]["value"], 9, "{v}");
        let (_, v) = send(&c, "POST", &coord.url("/s/_explain/1"), Some(json!({"query": {"match_all": {}}}))).await;
        assert_eq!(v["matched"], true, "{v}");
        let (_, v) = send(&c, "GET", &coord.url("/s/_termvectors/1?fields=t"), None).await;
        assert_eq!(v["found"], true, "{v}");
    }
    let after = (served(&c, i).await, served(&c, &search_node).await);
    assert_eq!(after.0, before.0, "the index node served no read");
    assert!(after.1.0 > before.1.0 && after.1.1 > before.1.1, "the search node served them: {before:?} -> {after:?}");
    // Realtime gets see unrefreshed writes; non-realtime gets do not until
    // a refresh (ES's stateless get: translog on the indexing shard for
    // realtime, the search shard's last refresh otherwise).
    let (st, _) = send(&c, "PUT", &i.url("/s/_doc/rt"), Some(json!({"t": "fresh"}))).await;
    assert_eq!(st, 201);
    let (_, v) = send(&c, "GET", &i.url("/s/_doc/rt"), None).await;
    assert_eq!(v["found"], true, "{v}");
    let (_, v) = send(&c, "GET", &i.url("/s/_doc/rt?realtime=false"), None).await;
    assert_eq!(v["found"], false, "{v}");
    let (_, v) = send(&c, "GET", &i.url("/s/_doc/rt?realtime=false&refresh=true"), None).await;
    assert_eq!(v["found"], true, "{v}");
    assert_eq!(served(&c, i).await.1, before.0.1, "gets never ran on the index node");
    // `solo` has no search copy: reads fail like an unassigned shard in ES.
    let (st, v) = send(&c, "GET", &i.url("/solo/_search"), None).await;
    assert_eq!(st, 503);
    assert_eq!(v, serde_json::from_str::<Value>(r#"{"error":{"root_cause":[{"type":"no_shard_available_action_exception","reason":null}],"type":"search_phase_execution_exception","reason":"all shards failed","phase":"query","grouped":true,"failed_shards":[{"shard":0,"index":"solo","node":null,"reason":{"type":"no_shard_available_action_exception","reason":null}}]},"status":503}"#).unwrap());
    let (st, v) = send(&c, "GET", &search_node.url("/solo/_doc/1"), None).await;
    assert_eq!((st, v["error"]["reason"].clone()), (503, json!("No shard available for [get [solo][1]: routing [null]]")));
    let (st, v) = send(&c, "GET", &i.url("/s,solo/_search?size=0"), None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["_shards"], json!({"total": 2, "successful": 1, "skipped": 0, "failed": 1, "failures": [{"shard": 0, "index": "solo", "node": null, "reason": {"type": "no_shard_available_action_exception", "reason": null}}]}));
    let (st, v) = send(&c, "GET", &i.url("/s,solo/_search?allow_partial_search_results=false"), None).await;
    assert_eq!((st, v["error"]["caused_by"]["reason"].clone()), (503, json!("Search rejected due to missing shards [[solo][0]]. Consider using `allow_partial_search_results` setting to bypass this error.")));
    // Writes to it still work: the indexing copy is fine.
    let (st, _) = send(&c, "PUT", &i.url("/solo/_doc/new"), Some(json!({"t": "x"}))).await;
    assert_eq!(st, 201);
    // Without the search node no shard has a search copy.
    search_node.kill();
    wait_nodes(&c, i, 1).await;
    let (st, v) = send(&c, "GET", &i.url("/m/_search"), None).await;
    assert_eq!((st, v["error"]["reason"].clone()), (503, json!("all shards failed")), "{v}");
    assert_eq!(v["error"]["failed_shards"].as_array().unwrap().len(), 1, "grouped per index");
    let (st, v) = send(&c, "GET", &i.url("/s/_doc/1"), None).await;
    assert_eq!((st, v["error"]["reason"].clone()), (503, json!("No shard available for [get [s][1]: routing [null]]")));
    let (st, v) = send(&c, "POST", &i.url("/_mget"), Some(json!({"docs": [{"_index": "m", "_id": "1"}]}))).await;
    assert_eq!(st, 200);
    assert_eq!(v["docs"][0]["error"]["reason"], "No shard available for [org.elasticsearch.action.get.MultiGetShardRequest/unset]", "{v}");
    let _ = index_node.log_text();
}

const CLUSTER_B: &str = "stateless-object-store";

/// A node of cluster B; `seeds`: the master's transport port (none for the
/// master itself).
async fn node_b(dir: &Path, seeds: &[u16], name: &str, roles: &str) -> Node {
    spawn(dir, Spec { cluster: CLUSTER_B, name, roles, seeds, masters: &["m"], extra: &[] }).await
}

/// Index one document, retrying until it is acknowledged (failovers make
/// some attempts fail; an acknowledged write must never be lost).
async fn put_acked(c: &reqwest::Client, n: &Node, id: usize) {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let r = c.put(n.url(&format!("/d/_doc/{id}"))).json(&json!({"n": id})).timeout(Duration::from_secs(70)).send().await;
        if let Ok(r) = r {
            if r.status().is_success() {
                return;
            }
        }
        assert!(Instant::now() < deadline, "write {id} never acknowledged");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

async fn primary_nodes(c: &reqwest::Client, n: &Node) -> Vec<String> {
    let (_, v) = send(c, "GET", &n.url("/_cat/shards/d?format=json&h=prirep,state,node"), None).await;
    v.as_array().into_iter().flatten().filter(|s| s["prirep"] == "p" && s["state"] == "STARTED").filter_map(|s| s["node"].as_str().map(str::to_string)).collect()
}

async fn count_all(c: &reqwest::Client, n: &Node) -> u64 {
    let _ = send(c, "POST", &n.url("/d/_refresh"), None).await;
    let (_, v) = send(c, "GET", &n.url("/d/_count"), None).await;
    v["count"].as_u64().unwrap_or(0)
}

/// Kill the index node(s) holding the primaries mid-indexing: another index
/// node recovers them from the object store and no acknowledged write is
/// lost; a search node rebuilt from scratch recovers from the object store;
/// finally a brand-new index node replaces the last one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn object_store_recovery_loses_no_acknowledged_write() {
    let dir = evidence::retained_dir("stateless-object-store-recovery");
    let d = dir.path();
    let master = node_b(d, &[], "m", "master,remote_cluster_client").await;
    let m = [master.transport];
    let mut i1 = node_b(d, &m, "i1", "index").await;
    let mut i2 = node_b(d, &m, "i2", "index").await;
    let mut s = node_b(d, &m, "s", "remote_cluster_client,search").await;
    let c = reqwest::Client::new();
    wait_nodes(&c, &master, 4).await;
    // No delayed allocation: the rebuilt search node is a new node.
    let (st, v) = send(&c, "PUT", &master.url("/d"), Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 1, "index.unassigned.node_left.delayed_timeout": "0"}}))).await;
    assert_eq!(st, 200, "{v}");
    wait_green(&c, &master, "d").await;
    let n = 300;
    for id in 0..n / 2 {
        put_acked(&c, &master, id).await;
    }
    // Kill whichever index node holds shard copies' primaries first.
    let holders = primary_nodes(&c, &master).await;
    let (victim, survivor) = if holders.iter().any(|h| h == "i1") { (&mut i1, "i2") } else { (&mut i2, "i1") };
    victim.kill();
    wait_nodes(&c, &master, 3).await;
    for id in n / 2..n {
        put_acked(&c, &master, id).await;
    }
    wait_green(&c, &master, "d").await;
    assert!(primary_nodes(&c, &master).await.iter().all(|h| h == survivor), "{:?}", primary_nodes(&c, &master).await);
    let surviving = if survivor == "i1" { &i1 } else { &i2 };
    assert!(surviving.log_text().contains("recovered the primary from the object store"), "primaries came back from the object store");
    assert_eq!(count_all(&c, &master).await, n as u64, "no acknowledged write lost after the index node failure");
    // Rebuild the search node from nothing: its copies come from the object
    // store (then the primary's translog above it).
    s.kill();
    wait_nodes(&c, &master, 2).await;
    std::fs::remove_dir_all(d.join("s")).unwrap();
    let s2 = node_b(d, &m, "s", "remote_cluster_client,search").await;
    wait_nodes(&c, &master, 3).await;
    wait_green(&c, &master, "d").await;
    assert!(s2.log_text().matches("recovered the search copy from the object store").count() >= 2, "{}", s2.log_text());
    assert_eq!(count_all(&c, &master).await, n as u64);
    // Replace the last index node with a brand-new one.
    drop(i1);
    drop(i2);
    wait_nodes(&c, &master, 2).await;
    let i3 = node_b(d, &m, "i3", "index").await;
    wait_nodes(&c, &master, 3).await;
    wait_green(&c, &master, "d").await;
    assert!(i3.log_text().contains("recovered the primary from the object store"));
    for id in n..n + 20 {
        put_acked(&c, &master, id).await;
    }
    assert_eq!(count_all(&c, &master).await, n as u64 + 20, "the new primaries hold every acknowledged write and take new ones");
    let ids: Vec<Value> = (0..n + 20).map(|i| json!({"_id": i.to_string()})).collect();
    let (_, v) = send(&c, "POST", &master.url("/d/_mget"), Some(json!({"docs": ids}))).await;
    let missing: Vec<&Value> = v["docs"].as_array().unwrap().iter().filter(|d| d["found"] != true).collect();
    assert!(missing.is_empty(), "missing: {missing:?}");
    // No index node at all: searches and non-realtime gets still run on the
    // search copies; realtime gets need the indexing shard and fail as in ES.
    drop(i3);
    wait_nodes(&c, &master, 2).await;
    let (st, v) = send(&c, "GET", &master.url("/d/_count"), None).await;
    assert_eq!((st, v["count"].clone()), (200, json!(n + 20)), "{v}");
    let (st, v) = send(&c, "GET", &master.url("/d/_doc/7?realtime=false"), None).await;
    assert_eq!((st, v["found"].clone()), (200, json!(true)), "{v}");
    let (st, v) = send(&c, "GET", &master.url("/d/_doc/7"), None).await;
    assert_eq!((st, v["error"]["type"].clone(), v["error"]["reason"].clone()), (503, json!("no_shard_available_action_exception"), json!("primary shard is not active")), "{v}");
    let (st, v) = send(&c, "POST", &master.url("/d/_mget"), Some(json!({"docs": [{"_id": "7"}]}))).await;
    assert_eq!((st, v["docs"][0]["error"]["reason"].clone()), (200, json!("primary shard is not active")), "{v}");
    let (_, v) = send(&c, "POST", &master.url("/d/_mget?realtime=false"), Some(json!({"docs": [{"_id": "7"}]}))).await;
    assert_eq!(v["docs"][0]["found"], true, "{v}");
    drop(s2);
}

const CLUSTER_C: &str = "stateless-field-ids";

/// Field ids follow the order fields first appear (`z` before `a`); a
/// search node rebuilt from nothing builds its index shell from the final
/// mapping, in sorted order. Its copy comes from the object store (commit
/// plus translog) and must answer every query exactly as the copy it
/// replaces did. Commits hold documents, re-parsed under the rebuilt shell's
/// own ids; this guards that path should commits ever ship segment files
/// (they would then have to take the commit's ids, as file recovery does).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rebuilt_search_copy_answers_a_dynamically_grown_mapping_as_before() {
    let dir = evidence::retained_dir("stateless-field-ids");
    let d = dir.path();
    let node = |name: &'static str, roles: &'static str, seeds: Vec<u16>| async move {
        spawn(d, Spec { cluster: CLUSTER_C, name, roles, seeds: &seeds, masters: &["m"], extra: &[] }).await
    };
    let master = node("m", "master,remote_cluster_client", vec![]).await;
    let _index = node("i", "index", vec![master.transport]).await;
    let mut search = node("s", "remote_cluster_client,search", vec![master.transport]).await;
    let c = reqwest::Client::new();
    wait_nodes(&c, &master, 3).await;
    let (st, v) = send(&c, "PUT", &master.url("/g"), Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 1, "index.unassigned.node_left.delayed_timeout": "0"}}))).await;
    assert_eq!(st, 200, "{v}");
    wait_green(&c, &master, "g").await;
    // One mapping change per request, in reverse name order.
    for (i, doc) in (0..60).map(|i| (i, json!({"z": i}))).chain((60..120).map(|i| (i, json!({"z": i, "a": format!("alpha w{}", i % 4)})))) {
        let (st, v) = send(&c, "PUT", &master.url(&format!("/g/_doc/{i}?refresh=true")), Some(doc)).await;
        assert!((200..300).contains(&st), "{v}");
    }
    let queries = [
        json!({"query": {"term": {"z": 7}}}),
        json!({"query": {"match": {"a": "w1"}}, "sort": [{"z": "asc"}], "size": 50}),
        json!({"query": {"range": {"z": {"gte": 100}}}, "sort": [{"z": "desc"}], "size": 50}),
        json!({"size": 0, "aggs": {"words": {"terms": {"field": "a.keyword"}}, "max": {"max": {"field": "z"}}}}),
    ];
    let answers = async |c: &reqwest::Client| {
        let mut out = Vec::new();
        for q in &queries {
            let (st, v) = send(c, "POST", &master.url("/g/_search"), Some(q.clone())).await;
            assert_eq!(st, 200, "{v}");
            out.push(json!({"hits": v["hits"]["hits"].as_array().unwrap().iter().map(|h| json!([h["_id"], h["_score"], h["sort"]])).collect::<Vec<_>>(),
                "total": v["hits"]["total"], "aggs": v["aggregations"]}));
        }
        out
    };
    let before = answers(&c).await;
    // Rebuild the search node from nothing.
    search.kill();
    wait_nodes(&c, &master, 2).await;
    std::fs::remove_dir_all(d.join("s")).unwrap();
    let search2 = node("s", "remote_cluster_client,search", vec![master.transport]).await;
    wait_nodes(&c, &master, 3).await;
    wait_green(&c, &master, "g").await;
    assert!(search2.log_text().contains("recovered the search copy from the object store"), "{}", search2.log_text());
    assert_eq!(answers(&c).await, before, "the rebuilt search copy answers as the one it replaced");
}
