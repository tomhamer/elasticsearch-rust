//! Dynamic mappings the ES way (Step 2 of the coordination redesign): the
//! master commits a new field before any copy indexes a document that
//! needs it; updates are batched and deduplicated on the master; a mapping
//! the master cannot commit in time fails only the documents that need it.

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/ports.rs"]
mod ports;

use elasticsearch_node::config::{NodeConfig, Settings};
use elasticsearch_node::node::ClusterNode;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A node on OS-assigned ports (read back with [`ports::http_port`]); with no
/// `masters` a single-node cluster, else a member that seeds from the
/// already-started nodes' transport ports `seeds` (none for the first).
fn settings(name: &str, cluster: &str, seeds: &[u16], masters: &[&str], extra: &[(&str, &str)]) -> Settings {
    let dir = evidence::scratch_dir(&format!("{cluster}-{name}"));
    let mut s = Settings::new();
    let mut put = |k: &str, v: Value| {
        s.insert(k.to_string(), v);
    };
    put("cluster.name", json!(cluster));
    put("node.name", json!(name));
    put("path.data", json!(dir.join("data").to_string_lossy()));
    put("path.logs", json!(dir.join("logs").to_string_lossy()));
    put("network.host", json!("127.0.0.1"));
    put("xpack.security.enabled", json!("false"));
    put("http.port", json!("0"));
    put("transport.port", json!("0"));
    if masters.is_empty() {
        put("discovery.type", json!("single-node"));
    } else {
        if !seeds.is_empty() {
            put("discovery.seed_hosts", json!(ports::seeds(seeds)));
        }
        put("cluster.initial_master_nodes", json!(masters));
    }
    for w in ["low", "high", "flood_stage"] {
        put(&format!("cluster.routing.allocation.disk.watermark.{w}"), json!("1b"));
    }
    for (k, v) in extra {
        put(k, json!(v));
    }
    s
}

async fn spawn_node(s: Settings) -> Arc<ClusterNode> {
    let cfg = NodeConfig::from_settings(s).expect("config");
    let (node, http) = ClusterNode::start(cfg).await.expect("node starts");
    elasticsearch_node::rest::register(&node);
    elasticsearch_node::snapshot_api::register(&node);
    let n = node.clone();
    tokio::spawn(async move {
        let _ = elasticsearch_node::front::serve(n, http).await;
    });
    node
}

async fn call(c: &reqwest::Client, http: u16, method: reqwest::Method, path: &str, body: Option<Value>) -> (u16, Value) {
    let mut r = c.request(method, format!("http://127.0.0.1:{http}{path}"));
    if let Some(b) = body {
        r = r.json(&b);
    }
    let Ok(resp) = r.send().await else { return (0, Value::Null) };
    let status = resp.status().as_u16();
    (status, resp.json().await.unwrap_or(Value::Null))
}

/// Start two master-eligible nodes and wait until one is the master and
/// both are members.
async fn two_nodes(cluster: &str, extra: &[(&str, &str)]) -> [Arc<ClusterNode>; 2] {
    let a = spawn_node(settings("n0", cluster, &[], &["n0", "n1"], extra)).await;
    let b = spawn_node(settings("n1", cluster, &[ports::transport_port(&a)], &["n0", "n1"], extra)).await;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let formed = [&a, &b].iter().all(|n| n.state().nodes.len() == 2 && n.state().master_node_id.is_some());
        if formed && (a.is_master() || b.is_master()) {
            return [a, b];
        }
        assert!(Instant::now() < deadline, "cluster did not form");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// The nodes' bound HTTP ports.
fn http_ports(nodes: &[Arc<ClusterNode>]) -> [u16; 2] {
    [ports::http_port(&nodes[0]), ports::http_port(&nodes[1])]
}

fn field_type(mapping: &Value, field: &str) -> Option<String> {
    mapping["properties"][field]["type"].as_str().map(str::to_string)
}

/// The mapping of `index` as each local engine holding it sees it.
fn local_mappings(node: &ClusterNode, index: &str) -> Vec<Value> {
    let mut out: Vec<Value> = node.engine.get_index(index).map(|s| s.mapping_json()).into_iter().collect();
    for (_, se) in node.shard_engines.all() {
        if let Some(s) = se.engine.get_index(index) {
            out.push(s.mapping_json());
        }
    }
    out
}

async fn create(c: &reqwest::Client, http: u16, index: &str, shards: u32, replicas: u32) {
    let (st, v) = call(c, http, reqwest::Method::PUT, &format!("/{index}"), Some(json!({"settings": {"number_of_shards": shards, "number_of_replicas": replicas}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(c, http, reqwest::Method::GET, &format!("/_cluster/health/{index}?wait_for_status=green&timeout=30s"), None).await;
    assert_eq!(st, 200, "{v}");
}

/// 200 documents, each introducing its own field, arrive at once on both
/// nodes. Every field is committed before its document is indexed, and the
/// master batches them: far fewer publications than fields.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_dynamic_mapping_wave_is_batched() {
    let nodes = two_nodes("mapping-wave", &[]).await;
    let http = http_ports(&nodes);
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let c = reqwest::Client::new();
    create(&c, http[0], "wave", 2, 1).await;
    let before = nodes[master].state().version;
    let writes = (0..200).map(|i| {
        let c = c.clone();
        let port = http[i % 2];
        async move { call(&c, port, reqwest::Method::PUT, &format!("/wave/_doc/{i}"), Some(json!({format!("f_{i}"): i}))).await }
    });
    for (st, v) in futures::future::join_all(writes).await {
        assert!((200..300).contains(&st), "{st} {v}");
    }
    let publications = nodes[master].state().version - before;
    assert!(publications <= 60, "200 new fields took {publications} publications: updates are not batched");
    let published = nodes[master].state().metadata.indices["wave"].mappings.clone();
    for i in 0..200 {
        assert_eq!(field_type(&published, &format!("f_{i}")).as_deref(), Some("long"), "field f_{i} committed");
    }
    // Every copy holds exactly the committed fields.
    for n in &nodes {
        for local in local_mappings(n, "wave") {
            for i in 0..200 {
                assert_eq!(field_type(&local, &format!("f_{i}")).as_deref(), Some("long"));
            }
        }
    }
}

/// Concurrent documents map the same new field with different types on
/// different primaries. The master resolves it once (first wins): one
/// committed type, every copy agrees with it, and the losing documents
/// fail instead of being indexed under a divergent local mapping.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_conflicting_dynamic_types_converge() {
    let nodes = two_nodes("mapping-conflict", &[]).await;
    let http = http_ports(&nodes);
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let c = reqwest::Client::new();
    create(&c, http[0], "conflict", 4, 0).await;
    let writes = (0..40).map(|i| {
        let c = c.clone();
        let port = http[i % 2];
        let value = if i % 2 == 0 { json!(7) } else { json!("seven") };
        async move { (i, call(&c, port, reqwest::Method::PUT, &format!("/conflict/_doc/{i}"), Some(json!({"x": value}))).await) }
    });
    let results = futures::future::join_all(writes).await;
    let published = nodes[master].state().metadata.indices["conflict"].mappings.clone();
    let ty = field_type(&published, "x").expect("x is mapped");
    for (i, (st, v)) in &results {
        let numeric = i % 2 == 0;
        match (ty.as_str(), numeric) {
            // Same type as the winner: its update merged or was already in.
            ("long", true) | ("text", false) => assert!((200..300).contains(st), "doc {i} has the committed type [{ty}]: {st} {v}"),
            // A string cannot be a long.
            ("long", false) => assert_eq!(*st, 400, "doc {i} conflicts with [{ty}]: {v}"),
            // A number fits a text field once the field is mapped, but its
            // own `long` proposal loses to a `text` committed first.
            ("text", true) => assert!((200..300).contains(st) || *st == 400, "doc {i}: {st} {v}"),
            (other, _) => panic!("unexpected type {other}"),
        }
    }
    for n in &nodes {
        assert_eq!(field_type(&n.state().metadata.indices["conflict"].mappings, "x").as_deref(), Some(ty.as_str()));
        for local in local_mappings(n, "conflict") {
            assert_eq!(field_type(&local, "x").as_deref(), Some(ty.as_str()), "a copy diverged from the committed mapping");
        }
    }
}

/// The master cannot commit mapping updates (its metadata service is held
/// up). A bulk on the other node: items with mapped fields index at once;
/// the item that needs a new field fails after `indices.mapping.dynamic_timeout`
/// with ES's error, and nothing indexes it under an uncommitted mapping.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn only_new_field_items_fail_when_the_master_cannot_commit() {
    let nodes = two_nodes("mapping-timeout", &[]).await;
    let http = http_ports(&nodes);
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let other = 1 - master;
    let c = reqwest::Client::new();
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"indices.mapping.dynamic_timeout": "2s"}}))).await;
    assert_eq!(st, 200, "{v}");
    let other_name = nodes[other].local.name.clone();
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/slow", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": other_name}, "mappings": {"properties": {"a": {"type": "long"}}}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(&c, http[master], reqwest::Method::GET, "/_cluster/health/slow?wait_for_status=green&timeout=30s", None).await;
    assert_eq!(st, 200, "{v}");
    for node in &nodes {
        assert_eq!(elasticsearch_node::mapping_update::dynamic_timeout(&node.state()), Duration::from_secs(2),
            "acknowledged cluster setting must be committed on every node before testing its timeout");
    }
    // Hold an actual service job. Dynamic mappings are now pure metadata
    // computations and do not acquire the live engine's metadata mutex.
    // Dropping `release` on any panic also unblocks the service worker.
    let (entered, ready) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel::<()>();
    assert!(nodes[master].metadata_service().submit(elasticsearch_node::master_service::Job::Pure {
        compute: Box::new(move |_, _, _| {
            let _ = entered.send(());
            let _ = wait.recv();
        }),
        done: Box::new(|_| {}),
    }));
    tokio::time::timeout(Duration::from_secs(10), ready).await.expect("blocking job starts").unwrap();
    let body = "{\"index\":{\"_index\":\"slow\",\"_id\":\"1\"}}\n{\"a\":1}\n{\"index\":{\"_index\":\"slow\",\"_id\":\"2\"}}\n{\"b\":\"new\"}\n";
    let started = Instant::now();
    let resp = c.post(format!("http://127.0.0.1:{}/_bulk", http[other])).header("content-type", "application/x-ndjson").body(body).send().await.expect("bulk");
    let elapsed = started.elapsed();
    let v: Value = resp.json().await.expect("bulk json");
    drop(release);
    assert!(elapsed >= Duration::from_millis(1500) && elapsed < Duration::from_secs(20), "bounded wait: {elapsed:?}");
    let items = v["items"].as_array().expect("items");
    assert!((200..300).contains(&items[0]["index"]["status"].as_u64().unwrap_or(0)), "a mapped item indexes: {v}");
    assert!(items[1]["index"]["error"].is_object(), "the new-field item fails: {v}");
    assert!(items[1]["index"]["status"].as_u64().unwrap_or(0) >= 400, "{v}");
    // Nothing indexed it locally under an uncommitted mapping.
    let (_, doc) = call(&c, http[other], reqwest::Method::GET, "/slow/_doc/2", None).await;
    assert_eq!(doc["found"], json!(false), "{doc}");
}

/// Review NH1: a document whose only new mapper is an object without leaf
/// fields (`{"meta": {}}`, `enabled: false`, `dynamic: false`, nested) does
/// not park the write for the dynamic timeout: the object never shows up
/// in the leaf field lookup, so it is not waited for.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn new_objects_without_fields_do_not_park_the_write() {
    let nodes = two_nodes("empty-objects", &[("cluster.auto_shrink_voting_configuration", "false")]).await;
    let http = http_ports(&nodes);
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let other = 1 - master;
    let c = reqwest::Client::new();
    let other_name = nodes[other].local.name.clone();
    // The primary on the non-master: the write goes through the master.
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/objs", Some(json!({
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.routing.allocation.require._name": other_name},
        "mappings": {"dynamic_templates": [
            {"nests": {"match": "nest*", "match_mapping_type": "object", "mapping": {"type": "nested"}}},
            {"offs": {"match": "off*", "match_mapping_type": "object", "mapping": {"type": "object", "enabled": false}}},
            {"looses": {"match": "loose*", "match_mapping_type": "object", "mapping": {"type": "object", "dynamic": false}}}
        ]}
    }))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(&c, http[master], reqwest::Method::GET, "/_cluster/health/objs?wait_for_status=green&timeout=30s", None).await;
    assert_eq!(st, 200, "{v}");
    for (i, doc) in [json!({"meta": {}}), json!({"off1": {"a": 1}}), json!({"loose1": {"x": 1}}), json!({"nest1": {}}), json!({"deep": {"inner": {}}})].into_iter().enumerate() {
        let started = Instant::now();
        let (st, v) = call(&c, http[other], reqwest::Method::PUT, &format!("/objs/_doc/{i}"), Some(doc.clone())).await;
        assert!((200..300).contains(&st), "{doc}: {st} {v}");
        assert!(started.elapsed() < Duration::from_secs(10), "{doc} parked for {:?}", started.elapsed());
    }
}

/// Review R3-B2: an index deleted while a wave of new-field writes waits
/// for its mapping commits. The delete fences the copies (it waits for the
/// writes' operation permits) while the applier holds its locks; a waiting
/// write must neither block on those locks nor outlive the fence. The
/// cluster keeps applying states afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn deleting_an_index_during_a_mapping_wave_does_not_stall_the_applier() {
    let nodes = two_nodes("wave-delete", &[]).await;
    let http = http_ports(&nodes);
    let master = if nodes[0].is_master() { 0 } else { 1 };
    let c = reqwest::Client::new();
    create(&c, http[0], "churn", 2, 1).await;
    // Writes after the delete must not re-create the index.
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"action.auto_create_index": "false"}}))).await;
    assert_eq!(st, 200, "{v}");
    // At most 32 writes in flight (each holds a connection on both ends;
    // hundreds at once can exhaust a test box's file descriptors).
    use futures::StreamExt;
    let wc = c.clone();
    let writes = futures::stream::iter(0..300usize).map(move |i| {
        let c = wc.clone();
        let port = http[i % 2];
        async move { call(&c, port, reqwest::Method::PUT, &format!("/churn/_doc/{i}"), Some(json!({format!("g_{i}"): i}))).await.0 }
    }).buffer_unordered(32).collect::<Vec<u16>>();
    let wave = tokio::spawn(writes);
    tokio::time::sleep(Duration::from_millis(100)).await;
    let (st, v) = call(&c, http[master], reqwest::Method::DELETE, "/churn", None).await;
    assert!(st == 200 || st == 404, "{st} {v}");
    let started = Instant::now();
    let statuses = tokio::time::timeout(Duration::from_secs(90), wave).await.expect("writes finished").unwrap();
    let unanswered = statuses.iter().filter(|s| **s == 0).count();
    assert_eq!(unanswered, 0, "{unanswered} writes got no answer; statuses: {statuses:?}");
    // The applier is alive: a new index commits and starts on both nodes.
    let (st, v) = call(&c, http[master], reqwest::Method::PUT, "/after-churn", Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 1}}))).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = call(&c, http[master], reqwest::Method::GET, "/_cluster/health/after-churn?wait_for_status=green&timeout=30s", None).await;
    assert_eq!(st, 200, "{v}");
    assert!(started.elapsed() < Duration::from_secs(60), "the cluster stalled for {:?}", started.elapsed());
    for n in &nodes {
        assert!(!n.state().metadata.indices.contains_key("churn"), "the delete was applied on [{}]", n.local.name);
        assert!(n.engine.get_index("churn").is_none(), "[{}] still holds a copy of the deleted index", n.local.name);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn constant_keyword_initialization_is_committed_to_every_copy() {
    let nodes = two_nodes("constant-initialization", &[]).await;
    let http = http_ports(&nodes);
    let c = reqwest::Client::new();
    create(&c, http[0], "constants", 2, 1).await;
    let (st, v) = call(&c, http[0], reqwest::Method::PUT, "/constants/_mapping", Some(json!({
        "properties":{"data_stream":{"properties":{"namespace":{"type":"constant_keyword"}}}}
    }))).await;
    assert_eq!(st, 200, "{v}");
    for i in 0..12 {
        let (st, v) = call(&c, http[i % 2], reqwest::Method::PUT, &format!("/constants/_doc/{i}?refresh=true"),
            Some(json!({"data_stream":{"namespace":"default"}, format!("dynamic_{i}"):i}))).await;
        assert_eq!(st, 201, "{v}");
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let all = nodes.iter().all(|n| {
            n.state().metadata.indices["constants"].mappings["properties"]["data_stream"]["properties"]["namespace"]["value"] == "default"
                && local_mappings(n, "constants").iter().all(|m| m["properties"]["data_stream"]["properties"]["namespace"]["value"] == "default")
        });
        if all { break; }
        assert!(Instant::now() < deadline, "constant did not converge across cluster metadata and copies");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let (st, v) = call(&c, http[1], reqwest::Method::PUT, "/constants/_doc/conflict", Some(json!({"data_stream":{"namespace":"other"}}))).await;
    assert_eq!(st, 400, "{v}");
    let (st, v) = call(&c, http[0], reqwest::Method::GET, "/constants/_count", None).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["count"], 12, "{v}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn conflicting_constant_first_values_have_one_committed_winner() {
    let nodes = two_nodes("constant-race", &[]).await;
    let http = http_ports(&nodes);
    let c = reqwest::Client::new();
    create(&c, http[0], "race", 2, 1).await;
    let (st, v) = call(&c, http[0], reqwest::Method::PUT, "/race/_mapping", Some(json!({"properties":{"c":{"type":"constant_keyword"}}}))).await;
    assert_eq!(st, 200, "{v}");
    let requests = (0..20).map(|i| {
        let c = c.clone();
        async move { call(&c, http[i % 2], reqwest::Method::PUT, &format!("/race/_doc/{i}?refresh=true"), Some(json!({"c":if i%2==0 {"a"} else {"b"}}))).await }
    });
    let results = futures::future::join_all(requests).await;
    let successes = results.iter().filter(|(st, _)| *st == 201).count();
    assert_eq!(successes, 10, "{results:?}");
    for (st,v) in &results {
        if *st != 201 {
            assert_eq!(*st, 400, "{v}");
            assert!(v.to_string().contains("only accepts values"), "{v}");
        }
    }
    let winner = nodes.iter().find(|n| n.is_master()).unwrap().state().metadata.indices["race"].mappings["properties"]["c"]["value"].clone();
    for n in &nodes { for m in local_mappings(n, "race") { assert_eq!(m["properties"]["c"]["value"], winner); } }
    let (_, count) = call(&c, http[0], reqwest::Method::GET, "/race/_count", None).await;
    assert_eq!(count["count"], 10);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn vector_dimensions_are_committed_before_indexing() {
    let nodes = two_nodes("vector-dims", &[]).await;
    let http = http_ports(&nodes);
    let c = reqwest::Client::new();
    create(&c, http[0], "vectors", 2, 1).await;
    let (st, v) = call(&c, http[0], reqwest::Method::PUT, "/vectors/_mapping", Some(json!({"properties":{"v":{"type":"dense_vector","index":false}}}))).await;
    assert_eq!(st, 200, "{v}");
    for i in 0..8 {
        let (st,v) = call(&c, http[i%2], reqwest::Method::PUT, &format!("/vectors/_doc/{i}?refresh=true"), Some(json!({"v":[1.0,2.0,3.0]}))).await;
        assert_eq!(st, 201, "{v}");
    }
    for n in &nodes {
        assert_eq!(n.state().metadata.indices["vectors"].mappings["properties"]["v"]["dims"], 3);
        for m in local_mappings(n,"vectors") { assert_eq!(m["properties"]["v"]["dims"],3); }
    }
    let (_, count) = call(&c, http[0], reqwest::Method::GET, "/vectors/_count", None).await;
    assert_eq!(count["count"],8);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn learned_parameters_land_in_normal_data_stream_not_failure_store() {
    let nodes = two_nodes("learned-datastream", &[]).await;
    let http = http_ports(&nodes);
    let c = reqwest::Client::new();
    let template = json!({"index_patterns":["traces-learned-*"],"priority":1000,"data_stream":{},"template":{
        "settings":{"number_of_shards":2,"number_of_replicas":1},
        "data_stream_options":{"failure_store":{"enabled":true}},
        "mappings":{"properties":{"@timestamp":{"type":"date"},"namespace":{"type":"constant_keyword"},"vector":{"type":"dense_vector","index":false}}}}});
    let (st,v) = call(&c,http[0],reqwest::Method::PUT,"/_index_template/learned",Some(template)).await;
    assert_eq!(st,200,"{v}");
    for i in 0..8 {
        let (st,v) = call(&c,http[i%2],reqwest::Method::PUT,&format!("/traces-learned-default/_create/{i}?refresh=true"),
            Some(json!({"@timestamp":"2026-10-23T00:00:00Z","namespace":"default","vector":[1.0,2.0,3.0],format!("new_{i}"):i}))).await;
        assert_eq!(st,201,"{v}");
        assert_ne!(v["failure_store"],"used","{v}");
        assert!(v["_index"].as_str().unwrap().starts_with(".ds-"),"{v}");
    }
    let (st,v) = call(&c,http[0],reqwest::Method::GET,"/traces-learned-default/_count",None).await;
    assert_eq!(st,200,"{v}"); assert_eq!(v["count"],8,"{v}");
    let (st,v) = call(&c,http[0],reqwest::Method::GET,"/traces-learned-default::failures/_count",None).await;
    assert_eq!(st,200,"{v}"); assert_eq!(v["count"],0,"{v}");
    for n in &nodes {
        for (name,im) in &n.state().metadata.indices {
            if name.starts_with(".ds-traces-learned-") {
                assert_eq!(im.mappings["properties"]["namespace"]["value"],"default");
                assert_eq!(im.mappings["properties"]["vector"]["dims"],3);
                for m in local_mappings(n,name) { assert_eq!(m, *im.mappings); }
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn persistent_replica_mapping_conflict_recovers_instead_of_retrying_forever() {
    let nodes = two_nodes("mapping-quarantine", &[]).await;
    let http = http_ports(&nodes);
    let c = reqwest::Client::new();
    create(&c,http[0],"quarantine",1,1).await;
    let (st,v) = call(&c,http[0],reqwest::Method::PUT,"/quarantine/_mapping",Some(json!({"properties":{"c":{"type":"constant_keyword"},"n":{"type":"long"}}}))).await;
    assert_eq!(st,200,"{v}");
    let (st,v) = call(&c,http[0],reqwest::Method::PUT,"/quarantine/_doc/retained?refresh=true",Some(json!({"n":1}))).await;
    assert_eq!(st,201,"{v}");
    let replica = nodes.iter().find(|n| !n.shards.is_primary("quarantine",0)).unwrap();
    let old = replica.engine_for(0).unwrap().get_index("quarantine").unwrap();
    old.put_mapping(&json!({"properties":{"c":{"type":"constant_keyword","value":"uncommitted"}}})).unwrap();
    let (st,v) = call(&c,http[0],reqwest::Method::PUT,"/quarantine/_mapping",Some(json!({"properties":{"added":{"type":"long"}}}))).await;
    assert_eq!(st,200,"{v}");
    let deadline = Instant::now()+Duration::from_secs(60);
    loop {
        let replaced = replica.engine_for(0).unwrap().get_index("quarantine").is_some_and(|svc| !Arc::ptr_eq(&svc, &old) && !svc.is_detached());
        let (_, h) = call(&c,http[0],reqwest::Method::GET,"/_cluster/health/quarantine",None).await;
        if replaced && h["status"] == "green" { break; }
        assert!(Instant::now()<deadline,"quarantined replica did not recover: {h}; routing: {:?}", replica.state().routing_table.shard_table("quarantine",0));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(old.is_detached());
    let (st,v) = call(&c,http[0],reqwest::Method::PUT,"/quarantine/_doc/new?refresh=true",Some(json!({"c":"committed","n":2}))).await;
    assert_eq!(st,201,"{v}");
    let (_,count)=call(&c,http[0],reqwest::Method::GET,"/quarantine/_count",None).await;
    assert_eq!(count["count"],2,"{count}");
}
