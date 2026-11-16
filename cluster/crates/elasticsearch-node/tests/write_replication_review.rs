//! Production review #89, #91, #92, #94 and #95 on a two-node in-process
//! cluster (every index has one primary and one replica, one per node):
//!
//! * #89: a document write marked for local execution (`x-es-cluster-exec-local`)
//!   still goes through the primary's write path, so the replica has every
//!   acknowledged operation: from a client (the edge strips the header) and
//!   as a forwarded request (the receiving node ignores the marker for writes);
//! * #91: `_source.includes` / `excludes` filter only the stored `_source`:
//!   the replica indexes the excluded fields as the primary does;
//! * #92: a `_source`-disabled index rebuilt from its documents
//!   (document-based peer recovery, a snapshot's document restore) keeps
//!   its indexed fields;
//! * #94: a partial update without `routing` keeps the stored routing;
//! * #95: an invalid `refresh` fails the request before anything changes.

mod common;

use common::{call, ports, settings, spawn_in_process, Ports};
use elasticsearch_node::config::NodeConfig;
use elasticsearch_node::node::ClusterNode;
use reqwest::Method;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

struct Two {
    c: reqwest::Client,
    p: Ports,
    nodes: Vec<Arc<ClusterNode>>,
    /// `path.repo` of both nodes.
    repo: String,
}

async fn start(extra: &[(&str, &str)]) -> Two {
    let cluster = ports::unique("write-review");
    let dir = common::data_dir(&cluster);
    let repo = dir.join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    let repo = repo.to_string_lossy().into_owned();
    let mut all: Vec<(&str, &str)> = vec![("path.repo", repo.as_str())];
    let repo = repo.clone();
    all.extend_from_slice(extra);
    let mut p = Ports { http: [0; 2], transport: [0; 2] };
    let mut nodes = Vec::new();
    for i in 0..2 {
        let cfg = NodeConfig::from_settings(settings(&cluster, p, i, &dir, &all)).expect("config");
        let node = spawn_in_process(cfg).await;
        p.http[i] = ports::http_port(&node);
        p.transport[i] = ports::transport_port(&node);
        nodes.push(node);
    }
    let c = reqwest::Client::new();
    ports::wait_formed(&c, &p.http, None, 2, Duration::from_secs(60)).await.unwrap_or_else(|e| panic!("cluster did not form: {e}"));
    Two { c, p, nodes, repo }
}

impl Two {
    async fn call(&self, node: usize, method: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        call(&self.c, self.p, node, method, path, body).await
    }

    async fn ok(&self, node: usize, method: Method, path: &str, body: Option<Value>) -> Value {
        let (st, v) = self.call(node, method.clone(), path, body).await;
        assert!((200..300).contains(&st), "{method} {path}: {st} {v}");
        v
    }

    async fn send(&self, node: usize, method: Method, path: &str, headers: &[(&str, &str)], ct: &str, body: String) -> (u16, Value) {
        let mut r = self.c.request(method, format!("http://127.0.0.1:{}{path}", self.p.http[node])).header("content-type", ct).body(body);
        for (k, v) in headers {
            r = r.header(*k, *v);
        }
        let resp = r.send().await.unwrap();
        let status = resp.status().as_u16();
        (status, resp.json().await.unwrap_or(Value::Null))
    }

    async fn green(&self, index: &str) {
        let (st, v) = self.call(0, Method::GET, &format!("/_cluster/health/{index}?wait_for_status=green&wait_for_no_initializing_shards=true&timeout=60s"), None).await;
        if v["status"] != "green" {
            let (_, explain) = self.call(0, Method::GET, "/_cluster/allocation/explain", Some(json!({"index": index.split(',').next().unwrap(), "shard": 0, "primary": false}))).await;
            panic!("[{index}] not green: {st} {v}\nexplain: {explain}");
        }
    }

    /// Position of the node holding the started primary of `[index][0]`.
    async fn primary(&self, index: &str) -> usize {
        let v = self.ok(0, Method::GET, &format!("/_cat/shards/{index}?format=json&h=prirep,state,node"), None).await;
        let node = v.as_array().unwrap().iter().find(|s| s["prirep"] == "p" && s["state"] == "STARTED").and_then(|s| s["node"].as_str()).unwrap_or_else(|| panic!("no primary: {v}")).to_string();
        if node == "n0" { 0 } else { 1 }
    }

    /// Hits of `query` on the copy node `node` holds (`preference=_local`).
    async fn local_hits(&self, node: usize, index: &str, query: Value) -> u64 {
        let v = self.ok(node, Method::POST, &format!("/{index}/_search?preference=_local&size=0&track_total_hits=true"), Some(json!({"query": query}))).await;
        assert_eq!(v["_shards"]["failed"], 0, "{v}");
        v["hits"]["total"]["value"].as_u64().unwrap()
    }
}

fn one_replica(mappings: Value) -> Value {
    json!({"settings": {"number_of_shards": 1, "number_of_replicas": 1}, "mappings": mappings})
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn exec_local_document_writes_are_replicated_before_acknowledgement() {
    let two = start(&[]).await;
    two.ok(0, Method::PUT, "/w", Some(one_replica(json!({"properties": {"k": {"type": "keyword"}}})))).await;
    two.green("w").await;
    let primary = two.primary("w").await;
    let replica = 1 - primary;
    let marker = [("x-es-cluster-exec-local", "1")];
    let mut expected = Vec::new();
    // From a client, through either node (local-primary control included):
    // single documents, creates, updates and bulks.
    for via in [replica, primary] {
        let id = format!("c{via}");
        let (st, v) = two.send(via, Method::PUT, &format!("/w/_doc/{id}"), &marker, "application/json", json!({"k": id}).to_string()).await;
        assert_eq!(st, 201, "{v}");
        expected.push(id.clone());
        let id = format!("cr{via}");
        let (st, v) = two.send(via, Method::PUT, &format!("/w/_create/{id}"), &marker, "application/json", json!({"k": "x"}).to_string()).await;
        assert_eq!(st, 201, "{v}");
        let (st, v) = two.send(via, Method::POST, &format!("/w/_update/{id}"), &marker, "application/json", json!({"doc": {"k": id}}).to_string()).await;
        assert_eq!(st, 200, "{v}");
        expected.push(id);
        let id = format!("b{via}");
        let bulk = format!("{}\n{}\n", json!({"index": {"_index": "w", "_id": id}}), json!({"k": id}));
        let (st, v) = two.send(via, Method::POST, "/_bulk", &marker, "application/x-ndjson", bulk).await;
        assert_eq!((st, &v["errors"]), (200, &json!(false)), "{v}");
        expected.push(id);
    }
    // As a forwarded request (hops > 0) marked for local execution on the
    // primary's node: what the edge strips from clients must not bypass
    // replication from inside the cluster either.
    let node = &two.nodes[primary];
    let forwarded = |method: &str, uri: &str, ct: &str, body: String| {
        elasticsearch_node::front::RestReq::try_new(method, uri, &[("content-type".into(), ct.into()), ("x-es-cluster-exec-local".into(), "1".into())], body.as_bytes(), 1, Default::default(), None).unwrap()
    };
    let r = elasticsearch_node::front::dispatch(node, forwarded("PUT", "/w/_doc/f1", "application/json", json!({"k": "f1"}).to_string())).await;
    assert_eq!(r.status, 201, "{:?}", r.body_json());
    expected.push("f1".into());
    let r = elasticsearch_node::front::dispatch(node, forwarded("POST", "/w/_update/f1", "application/json", json!({"doc": {"k": "f1u"}}).to_string())).await;
    assert_eq!(r.status, 200, "{:?}", r.body_json());
    let bulk = format!("{}\n{}\n", json!({"index": {"_index": "w", "_id": "f2"}}), json!({"k": "f2"}));
    let r = elasticsearch_node::front::dispatch(node, forwarded("POST", "/_bulk", "application/x-ndjson", bulk)).await;
    assert_eq!((r.status, r.body_json().map(|v| v["errors"].clone())), (200, Some(json!(false))), "{:?}", r.body_json());
    expected.push("f2".into());
    two.ok(0, Method::POST, "/w/_refresh", None).await;
    // Every acknowledged operation is on the replica: losing the primary
    // right now would lose none of them.
    for id in &expected {
        let want = if id == "f1" { "f1u" } else { id.as_str() };
        assert_eq!(two.local_hits(replica, "w", json!({"term": {"k": want}})).await, 1, "[{id}] acknowledged but missing on the replica");
    }
    assert_eq!(two.local_hits(replica, "w", json!({"match_all": {}})).await, expected.len() as u64);
    assert_eq!(two.local_hits(primary, "w", json!({"match_all": {}})).await, expected.len() as u64);
    // The replica's history matches the primary's (same seq_nos acknowledged).
    let stats = two.ok(0, Method::GET, "/w/_stats?level=shards", None).await;
    let copies = stats["indices"]["w"]["shards"]["0"].as_array().unwrap();
    let max: Vec<&Value> = copies.iter().map(|c| &c["seq_no"]["max_seq_no"]).collect();
    assert_eq!(max[0], max[1], "{stats}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn source_filtering_keeps_indexed_fields_on_the_replica() {
    let two = start(&[]).await;
    let mappings = [
        ("excl", json!({"_source": {"excludes": ["secret", "nested.hidden"]}, "properties": {"secret": {"type": "keyword"}, "nested": {"properties": {"hidden": {"type": "long"}}}}})),
        ("incl", json!({"_source": {"includes": ["open"]}, "properties": {"secret": {"type": "keyword"}, "open": {"type": "keyword"}, "nested": {"properties": {"hidden": {"type": "long"}}}}})),
    ];
    for (index, m) in &mappings {
        two.ok(0, Method::PUT, &format!("/{index}"), Some(one_replica(m.clone()))).await;
        two.green(index).await;
        let primary = two.primary(index).await;
        // Single-document and bulk writes, through the replica's node and the primary's.
        let (st, v) = two.call(1 - primary, Method::PUT, &format!("/{index}/_doc/1"), Some(json!({"secret": "s1", "open": "o1", "nested": {"hidden": 7}}))).await;
        assert_eq!(st, 201, "{v}");
        let bulk = (2..6).map(|i| format!("{}\n{}\n", json!({"index": {"_index": index, "_id": i.to_string()}}), json!({"secret": format!("s{i}"), "open": format!("o{i}"), "nested": {"hidden": i}}))).collect::<String>();
        let (st, v) = two.send(primary, Method::POST, "/_bulk", &[], "application/x-ndjson", bulk).await;
        assert_eq!((st, &v["errors"]), (200, &json!(false)), "{v}");
        two.ok(0, Method::POST, &format!("/{index}/_refresh"), None).await;
        for node in [primary, 1 - primary] {
            for (q, want) in [(json!({"term": {"secret": "s1"}}), 1), (json!({"term": {"secret": "s4"}}), 1), (json!({"range": {"nested.hidden": {"gte": 4}}}), 3)] {
                assert_eq!(two.local_hits(node, index, q.clone()).await, want, "[{index}] {q} on {}", if node == primary { "the primary" } else { "the replica" });
            }
            // Doc values of an excluded field agree too.
            let v = two.ok(node, Method::POST, &format!("/{index}/_search?preference=_local&size=0"), Some(json!({"aggs": {"m": {"max": {"field": "nested.hidden"}}}}))).await;
            assert_eq!(v["aggregations"]["m"]["value"], 7.0, "{v}");
        }
        // The stored `_source` stays filtered (realtime and after refresh).
        for refresh in ["", "?realtime=false"] {
            let v = two.ok(0, Method::GET, &format!("/{index}/_doc/1{refresh}"), None).await;
            assert_eq!(v["_source"]["open"], "o1", "{v}");
            assert!(v["_source"].get("secret").is_none() && v["_source"]["nested"].get("hidden").is_none(), "{v}");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn document_based_recovery_and_restore_keep_source_disabled_data() {
    // Document-based peer recovery and document restore (the fallbacks
    // `indices.recovery.file_based` / `snapshot.restore.file_based` select).
    let two = start(&[("indices.recovery.file_based", "false"), ("snapshot.restore.file_based", "false")]).await;
    for (index, m) in [
        ("disabled", json!({"_source": {"enabled": false}, "properties": {"k": {"type": "keyword"}, "n": {"type": "long"}}})),
        ("excluded", json!({"_source": {"excludes": ["k"]}, "properties": {"k": {"type": "keyword"}, "n": {"type": "long"}}})),
    ] {
        two.ok(0, Method::PUT, &format!("/{index}"), Some(json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": m}))).await;
        let bulk = (0..20).map(|i| format!("{}\n{}\n", json!({"index": {"_index": index, "_id": i.to_string()}}), json!({"k": format!("v{}", i % 4), "n": i}))).collect::<String>();
        let (st, v) = two.send(0, Method::POST, "/_bulk?refresh=true", &[], "application/x-ndjson", bulk).await;
        assert_eq!((st, &v["errors"]), (200, &json!(false)), "{v}");
        // Flushed: the copy is rebuilt from its documents, not its translog.
        two.ok(0, Method::POST, &format!("/{index}/_flush"), None).await;
        let primary = two.primary(index).await;
        two.ok(0, Method::PUT, &format!("/{index}/_settings"), Some(json!({"index": {"number_of_replicas": 1}}))).await;
        two.green(index).await;
        let queries = [(json!({"term": {"k": "v1"}}), 5), (json!({"range": {"n": {"gte": 15}}}), 5), (json!({"match_all": {}}), 20)];
        for (q, want) in &queries {
            assert_eq!(two.local_hits(primary, index, q.clone()).await, *want, "[{index}] primary {q}");
            assert_eq!(two.local_hits(1 - primary, index, q.clone()).await, *want, "[{index}] recovered replica {q}");
        }
        let v = two.ok(1 - primary, Method::POST, &format!("/{index}/_search?preference=_local&size=0"), Some(json!({"aggs": {"m": {"max": {"field": "n"}}}}))).await;
        assert_eq!(v["aggregations"]["m"]["value"], 19.0, "{v}");
    }
    // Snapshot, then restore from the documents.
    two.ok(0, Method::PUT, "/_snapshot/r", Some(json!({"type": "fs", "settings": {"location": two.repo.clone()}}))).await;
    two.ok(0, Method::PUT, "/_snapshot/r/s?wait_for_completion=true", Some(json!({"indices": "disabled,excluded"}))).await;
    two.ok(0, Method::DELETE, "/disabled,excluded", None).await;
    let v = two.ok(0, Method::POST, "/_snapshot/r/s/_restore?wait_for_completion=true", Some(json!({"indices": "disabled,excluded"}))).await;
    assert_eq!(v["snapshot"]["shards"]["failed"], 0, "{v}");
    for index in ["disabled", "excluded"] {
        two.green(index).await;
        two.ok(0, Method::POST, &format!("/{index}/_refresh"), None).await;
        for node in [0, 1] {
            assert_eq!(two.local_hits(node, index, json!({"term": {"k": "v2"}})).await, 5, "[{index}] restored on n{node}");
            assert_eq!(two.local_hits(node, index, json!({"range": {"n": {"lt": 3}}})).await, 3, "[{index}] restored on n{node}");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn partial_updates_keep_stored_routing_and_invalid_refresh_changes_nothing() {
    let two = start(&[]).await;
    // #94: one shard, optional routing.
    two.ok(0, Method::PUT, "/r", Some(one_replica(json!({"properties": {"n": {"type": "long"}}})))).await;
    two.green("r").await;
    two.ok(0, Method::PUT, "/r/_doc/1?routing=rt", Some(json!({"n": 1}))).await;
    two.ok(1, Method::PUT, "/r/_doc/2?routing=rt", Some(json!({"n": 1}))).await;
    let v = two.ok(0, Method::POST, "/r/_update/1", Some(json!({"doc": {"n": 2}}))).await;
    assert_eq!(v["result"], "updated", "{v}");
    let bulk = format!("{}\n{}\n", json!({"update": {"_index": "r", "_id": "2"}}), json!({"doc": {"n": 2}}));
    let (st, v) = two.send(1, Method::POST, "/_bulk?refresh=true", &[], "application/x-ndjson", bulk).await;
    assert_eq!((st, &v["errors"]), (200, &json!(false)), "{v}");
    for node in [0, 1] {
        let v = two.ok(node, Method::POST, "/r/_search?preference=_local", Some(json!({"query": {"term": {"_routing": "rt"}}}))).await;
        assert_eq!(v["hits"]["total"]["value"], 2, "routing kept on n{node}: {v}");
        for id in ["1", "2"] {
            let v = two.ok(node, Method::GET, &format!("/r/_doc/{id}?routing=rt"), None).await;
            assert_eq!((&v["_routing"], &v["_source"]["n"]), (&json!("rt"), &json!(2)), "{v}");
        }
    }

    // #95: an invalid `refresh` changes nothing, whatever the write.
    let bad = "refresh=typo";
    let unknown = |v: &Value| v["error"]["type"] == "illegal_argument_exception" && v["error"]["reason"] == "Unknown value for refresh: [typo].";
    let (st, v) = two.call(0, Method::PUT, &format!("/autocreated/_doc/1?{bad}"), Some(json!({"n": 1}))).await;
    assert!(st == 400 && unknown(&v), "{st} {v}");
    let (st, _) = two.call(0, Method::GET, "/autocreated", None).await;
    assert_eq!(st, 404, "an invalid refresh must not auto-create the index");
    two.ok(0, Method::PUT, "/s/_doc/1?refresh=true", Some(json!({"n": 0}))).await;
    two.green("s").await;
    let before = two.ok(0, Method::GET, "/s/_doc/1", None).await;
    let script = json!({"script": {"source": "ctx._source.n += 1"}});
    let bulk = format!("{}\n{}\n{}\n", json!({"update": {"_index": "s", "_id": "1"}}), script, json!({"delete": {"_index": "s", "_id": "1"}}));
    for via in [0, 1] {
        for (method, path, body) in [
            (Method::PUT, format!("/s/_doc/1?{bad}"), Some(json!({"n": 100}))),
            (Method::PUT, format!("/s/_create/2?{bad}"), Some(json!({"n": 100}))),
            (Method::POST, format!("/s/_doc?{bad}"), Some(json!({"n": 100}))),
            (Method::DELETE, format!("/s/_doc/1?{bad}"), None),
            (Method::POST, format!("/s/_update/1?{bad}"), Some(script.clone())),
        ] {
            let (st, v) = two.call(via, method.clone(), &path, body).await;
            assert!(st == 400 && unknown(&v), "{method} {path} via n{via}: {st} {v}");
        }
        let (st, v) = two.send(via, Method::POST, &format!("/_bulk?{bad}"), &[], "application/x-ndjson", bulk.clone()).await;
        assert!(st == 400 && unknown(&v), "bulk via n{via}: {st} {v}");
    }
    two.ok(0, Method::POST, "/s/_refresh", None).await;
    let after = two.ok(0, Method::GET, "/s/_doc/1", None).await;
    for k in ["_version", "_seq_no", "_primary_term", "_source"] {
        assert_eq!(after[k], before[k], "{k}: {before} -> {after}");
    }
    let v = two.ok(0, Method::POST, "/s/_count", None).await;
    assert_eq!(v["count"], 1, "nothing created: {v}");
    // Corrected and retried: the script has run exactly once in total.
    two.ok(1, Method::POST, "/s/_update/1?refresh=true", Some(script)).await;
    // Two operations in total: the first index and this update.
    for node in [0, 1] {
        let v = two.ok(node, Method::GET, "/s/_doc/1?preference=_local", None).await;
        assert_eq!((&v["_source"]["n"], &v["_seq_no"], &v["_version"]), (&json!(1), &json!(1), &json!(2)), "{v}");
    }
}
