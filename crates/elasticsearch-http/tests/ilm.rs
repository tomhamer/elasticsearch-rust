//! HTTP tests for index lifecycle features: ILM policies, explain, move /
//! retry / remove, the background executor (rollover, delete), resize
//! (`_clone` / `_shrink` / `_split`), data stream lifecycle and the data
//! stream management APIs, SLM, and `_freeze` / `_unfreeze`.

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::time::Duration;

struct Es {
    base: String,
    c: Client,
}

async fn start(dir: &std::path::Path) -> Es {
    let node = Node::open(dir).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    Es { base: format!("http://{addr}"), c: Client::new() }
}

impl Es {
    async fn call(&self, m: Method, p: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.c.request(m, format!("{}{}", self.base, p));
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b.to_string());
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let t = resp.text().await.unwrap();
        (st, if t.is_empty() { Value::Null } else { serde_json::from_str(&t).unwrap_or(Value::String(t)) })
    }
    async fn get(&self, p: &str) -> (u16, Value) {
        self.call(Method::GET, p, None).await
    }
    async fn post(&self, p: &str, b: Value) -> (u16, Value) {
        self.call(Method::POST, p, Some(b)).await
    }
    async fn put(&self, p: &str, b: Value) -> (u16, Value) {
        self.call(Method::PUT, p, Some(b)).await
    }
    async fn delete(&self, p: &str) -> (u16, Value) {
        self.call(Method::DELETE, p, None).await
    }
    async fn ok(&self, m: Method, p: &str, b: Option<Value>) -> Value {
        let (st, v) = self.call(m, p, b).await;
        assert!(st == 200 || st == 201, "{p}: {st} {v}");
        v
    }
    /// Poll `GET p` until `pred` holds (background executor).
    async fn wait_for(&self, p: &str, pred: impl Fn(u16, &Value) -> bool) -> Value {
        for _ in 0..60 {
            let (st, v) = self.get(p).await;
            if pred(st, &v) {
                return v;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        let (_, v) = self.get(p).await;
        panic!("condition not reached for {p}: {v}");
    }
}

fn warm_delete_policy() -> Value {
    json!({"policy": {"phases": {
        "warm": {"min_age": "10s", "actions": {"forcemerge": {"max_num_segments": 10000}}},
        "delete": {"min_age": "30s", "actions": {"delete": {}}}
    }}})
}

#[tokio::test]
async fn policy_crud_and_validation() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, _) = es.get("/_ilm/policy/p1").await;
    assert_eq!(st, 404);
    let (st, v) = es.delete("/_ilm/policy/p1").await;
    assert_eq!((st, v["error"]["type"].as_str()), (404, Some("resource_not_found_exception")));
    assert_eq!(es.ok(Method::PUT, "/_ilm/policy/p1", Some(warm_delete_policy())).await, json!({"acknowledged": true}));
    let v = es.ok(Method::GET, "/_ilm/policy/p1", None).await;
    assert_eq!(v["p1"]["version"], 1);
    assert!(v["p1"]["modified_date"].as_str().unwrap().ends_with('Z'));
    assert_eq!(v["p1"]["policy"]["phases"]["warm"]["min_age"], "10s");
    assert_eq!(v["p1"]["policy"]["phases"]["delete"]["actions"]["delete"]["delete_searchable_snapshot"], true);
    assert_eq!(v["p1"]["in_use_by"], json!({"indices": [], "data_streams": [], "composable_templates": []}));
    // identical PUT is a no-op, a change bumps the version
    es.ok(Method::PUT, "/_ilm/policy/p1", Some(warm_delete_policy())).await;
    assert_eq!(es.ok(Method::GET, "/_ilm/policy/p1", None).await["p1"]["version"], 1);
    es.ok(Method::PUT, "/_ilm/policy/p1", Some(json!({"policy": {"phases": {"warm": {"min_age": "300s", "actions": {}}}}}))).await;
    assert_eq!(es.ok(Method::GET, "/_ilm/policy/p1", None).await["p1"]["version"], 2);
    es.ok(Method::PUT, "/_ilm/policy/p2", Some(warm_delete_policy())).await;
    let v = es.ok(Method::GET, "/_ilm/policy/p*", None).await;
    assert!(v.get("p1").is_some() && v.get("p2").is_some());
    let v = es.ok(Method::GET, "/_ilm/policy", None).await;
    assert!(v.get("logs@lifecycle").is_some(), "built-in policies installed");
    let (st, v) = es.get("/_ilm/policy/p*,p2").await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["reason"], "wildcard only supports a single value, please use comma-separated values or a single wildcard value");
    let (st, _) = es.get("/_ilm/policy/p1,nope").await;
    assert_eq!(st, 404);
    // validation
    for bad in [
        json!({"policy": {"phases": {"hot": {"actions": {"forcemerge": {"max_num_segments": 1}}}}}}),
        json!({"policy": {"phases": {"hot": {"actions": {"shrink": {"number_of_shards": 1}}}}}}),
        json!({"policy": {"phases": {"hot": {"actions": {"readonly": {}}}}}}),
        json!({"policy": {"phases": {"hot": {"actions": {"delete": {}}}}}}),
        json!({"policy": {"phases": {"hot": {"actions": {"foo": {}}}}}}),
        json!({"policy": {"phases": {"hot": {"actions": {"rollover": {"min_docs": 1}}}}}}),
        json!({"policy": {"phases": {"delete": {"actions": {"wait_for_snapshot": {"policy": "missing"}}}}}}),
    ] {
        let (st, v) = es.put("/_ilm/policy/bad", bad.clone()).await;
        assert_eq!(st, 400, "{bad} -> {v}");
    }
    // `searchable_snapshot` needs an enterprise license, which rustic always
    // has (AGENTS.md "Licensing: always enterprise"): the policy is accepted.
    let ss = json!({"policy": {"phases": {"cold": {"actions": {"searchable_snapshot": {"snapshot_repository": "r"}}}}}});
    let (st, v) = es.put("/_ilm/policy/searchable", ss).await;
    assert_eq!(st, 200, "{v}");
    let (st, _) = es.call(Method::DELETE, "/_ilm/policy/searchable", None).await;
    assert_eq!(st, 200);
    let (st, v) = es.put("/_ilm/policy/bad", json!({"policy": {"phases": {"warm": {"min_age": "10s", "actions": {}}, "delete": {"min_age": "5s", "actions": {"delete": {}}}}}})).await;
    assert_eq!(st, 400);
    assert!(v["error"]["reason"].as_str().unwrap().contains("Your policy is configured to run the delete phase (min_age: 5s) before the warm phase (min_age: 10s)"), "{v}");
    let (st, v) = es
        .put("/_ilm/policy/bad", json!({"policy": {"phases": {"warm": {"min_age": "10s", "actions": {"downsample": {"fixed_interval": "3h"}}}, "cold": {"min_age": "30s", "actions": {"downsample": {"fixed_interval": "3h"}}}}}}))
        .await;
    assert_eq!(st, 400);
    assert!(v["error"]["reason"].as_str().unwrap().contains("Downsampling interval [3h] for phase [cold] must be greater than the interval [3h] for phase [warm]"));
    // in use → cannot delete
    es.ok(Method::PUT, "/idx", Some(json!({"settings": {"index.lifecycle.name": "p2"}}))).await;
    let (st, v) = es.delete("/_ilm/policy/p2").await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["root_cause"][0]["reason"], "Cannot delete policy [p2]. It is in use by one or more indices: [idx]");
    assert_eq!(es.ok(Method::GET, "/_ilm/policy/p2", None).await["p2"]["in_use_by"]["indices"], json!(["idx"]));
    let v = es.ok(Method::POST, "/idx/_ilm/remove", None).await;
    assert_eq!(v, json!({"has_failures": false, "failed_indexes": []}));
    let v = es.ok(Method::GET, "/idx/_settings", None).await;
    assert!(v["idx"]["settings"]["index"].get("lifecycle").is_none(), "{v}");
    es.ok(Method::DELETE, "/_ilm/policy/p2", None).await;
    let (st, _) = es.call(Method::POST, "/missing/_ilm/remove", None).await;
    assert_eq!(st, 404);
    // status / stop / start
    assert_eq!(es.ok(Method::GET, "/_ilm/status", None).await, json!({"operation_mode": "RUNNING"}));
    es.ok(Method::POST, "/_ilm/stop", None).await;
    let m = es.ok(Method::GET, "/_ilm/status", None).await["operation_mode"].as_str().unwrap().to_string();
    assert!(m == "STOPPING" || m == "STOPPED");
    es.wait_for("/_ilm/status", |_, v| v["operation_mode"] == "STOPPED").await;
    es.ok(Method::POST, "/_ilm/start", None).await;
    assert_eq!(es.ok(Method::GET, "/_ilm/status", None).await["operation_mode"], "RUNNING");
    let v = es.ok(Method::GET, "/_health_report/ilm", None).await;
    assert_eq!(v["indicators"]["ilm"]["details"]["ilm_status"], "RUNNING");
    assert!(v["indicators"]["ilm"]["details"]["policies"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn explain_move_retry() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.ok(
        Method::PUT,
        "/_ilm/policy/mv",
        Some(json!({"policy": {"phases": {"warm": {"min_age": "1000s", "actions": {"forcemerge": {"max_num_segments": 10000}}}, "hot": {"min_age": "1000s", "actions": {}}}}})),
    )
    .await;
    es.ok(Method::PUT, "/my_index", Some(json!({"settings": {"index.lifecycle.name": "mv"}}))).await;
    es.ok(Method::PUT, "/unmanaged", None).await;
    es.ok(Method::PUT, "/bad_policy_index", Some(json!({"settings": {"index.lifecycle.name": "does_not_exist"}}))).await;
    es.ok(Method::PUT, "/hidden_index", Some(json!({"settings": {"index.hidden": true, "index.lifecycle.name": "mv"}}))).await;

    let v = es.ok(Method::GET, "/my_index/_ilm/explain", None).await;
    let e = &v["indices"]["my_index"];
    assert_eq!((e["phase"].as_str(), e["action"].as_str(), e["step"].as_str()), (Some("new"), Some("complete"), Some("complete")), "{v}");
    assert_eq!(e["managed"], true);
    assert_eq!(e["policy"], "mv");
    assert!(e["age"].is_string() && e["time_since_index_creation"].is_string());
    assert_eq!(e["phase_execution"]["policy"], "mv");
    assert_eq!(e["phase_execution"]["version"], 1);
    assert_eq!(e["skip"], false);
    let v = es.ok(Method::GET, "/unmanaged/_ilm/explain", None).await;
    assert_eq!(v["indices"], json!({"unmanaged": {"index": "unmanaged", "managed": false}}));
    let v = es.ok(Method::GET, "/*/_ilm/explain?only_managed=true", None).await;
    let names: Vec<&String> = v["indices"].as_object().unwrap().keys().collect();
    assert_eq!(names, vec!["bad_policy_index", "my_index"]);
    let v = es.ok(Method::GET, "/*/_ilm/explain?only_errors=true", None).await;
    let e = &v["indices"]["bad_policy_index"];
    assert_eq!(e["step"], "ERROR");
    assert_eq!(e["step_info"]["reason"], "policy [does_not_exist] does not exist");
    assert_eq!(v["indices"].as_object().unwrap().len(), 1);
    let v = es.ok(Method::GET, "/hidden_*/_ilm/explain?expand_wildcards=open,hidden&human", None).await;
    assert!(v["indices"]["hidden_index"]["index_creation_date"].as_str().unwrap().ends_with('Z'), "{v}");

    // move to step
    let (st, v) = es
        .post(
            "/_ilm/move/my_index",
            json!({"current_step": {"phase": "warm", "action": "forcemerge", "name": "forcemerge"}, "next_step": {"phase": "warm", "action": "forcemerge", "name": "forcemerge"}}),
        )
        .await;
    assert_eq!(st, 400);
    assert_eq!(
        v["error"]["root_cause"][0]["reason"],
        "index [my_index] is not on current step [{\"phase\":\"warm\",\"action\":\"forcemerge\",\"name\":\"forcemerge\"}], currently: [{\"phase\":\"new\",\"action\":\"complete\",\"name\":\"complete\"}]"
    );
    let (_, v) = es.post("/_ilm/move/unmanaged", json!({"current_step": {"phase": "hot", "action": "x", "name": "y"}, "next_step": {"phase": "warm"}})).await;
    assert_eq!(v["error"]["root_cause"][0]["reason"], "index [unmanaged] is not associated with an Index Lifecycle Policy");
    let (_, v) = es.post("/_ilm/move/nope", json!({"current_step": {"phase": "hot", "action": "x", "name": "y"}, "next_step": {"phase": "warm"}})).await;
    assert_eq!(v["error"]["root_cause"][0]["reason"], "index [nope] does not exist");
    // a valid move into the warm phase runs the forcemerge action
    es.ok(
        Method::POST,
        "/_ilm/move/my_index",
        Some(json!({"current_step": {"phase": "new", "action": "complete", "name": "complete"}, "next_step": {"phase": "warm", "action": "forcemerge"}})),
    )
    .await;
    let v = es.ok(Method::GET, "/my_index/_ilm/explain", None).await;
    let e = &v["indices"]["my_index"];
    assert_eq!(e["phase"], "warm");
    assert_eq!(e["phase_execution"]["phase_definition"]["actions"]["forcemerge"]["max_num_segments"], 10000);
    // forcemerge made the index read-only and completed the phase (the
    // injected migrate action runs before forcemerge in warm)
    assert_eq!((e["action"].as_str(), e["step"].as_str()), (Some("complete"), Some("complete")), "{v}");
    let s = es.ok(Method::GET, "/my_index/_settings", None).await;
    assert_eq!(s["my_index"]["settings"]["index"]["blocks"]["write"], "true");
    // moving to the warm phase itself starts with the injected migrate, which
    // waits for replicas a single node cannot allocate
    es.ok(Method::PUT, "/other", Some(json!({"settings": {"index.lifecycle.name": "mv"}}))).await;
    es.ok(Method::GET, "/other/_ilm/explain", None).await;
    es.ok(
        Method::POST,
        "/_ilm/move/other",
        Some(json!({"current_step": {"phase": "new", "action": "complete", "name": "complete"}, "next_step": {"phase": "warm"}})),
    )
    .await;
    let v = es.ok(Method::GET, "/other/_ilm/explain", None).await;
    let e = &v["indices"]["other"];
    assert_eq!((e["action"].as_str(), e["step"].as_str()), (Some("migrate"), Some("check-migration")), "{v}");
    assert_eq!(e["step_info"]["message"], "Waiting for all shard copies to be active");
    let s = es.ok(Method::GET, "/other/_settings", None).await;
    assert_eq!(s["other"]["settings"]["index"]["routing"]["allocation"]["include"]["_tier_preference"], "data_warm,data_hot");

    // retry
    let (st, v) = es.call(Method::POST, "/my_index/_ilm/retry", None).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["root_cause"][0]["reason"], "cannot retry an action for an index [my_index] that has not encountered an error when running a Lifecycle Policy");
    let (_, v) = es.call(Method::POST, "/does_not_exist/_ilm/retry", None).await;
    assert_eq!(v["error"]["root_cause"][0]["reason"], "index [does_not_exist] does not exist");
}

#[tokio::test]
async fn executor_rollover_and_delete() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.ok(Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"indices.lifecycle.poll_interval": "1s"}}))).await;
    es.ok(
        Method::PUT,
        "/_ilm/policy/roll",
        Some(json!({"policy": {"phases": {"hot": {"actions": {"rollover": {"max_docs": 1}, "set_priority": {"priority": 50}}}, "delete": {"min_age": "1h", "actions": {"delete": {}}}}}})),
    )
    .await;
    es.ok(Method::PUT, "/logs-000001", Some(json!({"settings": {"index.lifecycle.name": "roll", "index.lifecycle.rollover_alias": "logs", "number_of_replicas": 0}, "aliases": {"logs": {"is_write_index": true}}}))).await;
    let v = es.ok(Method::GET, "/logs-000001/_ilm/explain", None).await;
    assert_eq!(v["indices"]["logs-000001"]["phase"], "hot");
    assert_eq!(es.ok(Method::GET, "/logs-000001/_settings", None).await["logs-000001"]["settings"]["index"]["priority"], "50");
    es.ok(Method::POST, "/logs/_doc?refresh=true", Some(json!({"m": 1}))).await;
    let v = es
        .wait_for("/logs-000001/_ilm/explain", |_, v| {
            let e = &v["indices"]["logs-000001"];
            e["action"] == "complete" && e["phase"] == "hot"
        })
        .await;
    assert!(v["indices"]["logs-000001"]["lifecycle_date_millis"].as_i64().unwrap() > 0);
    // rolled over: the new index is the write index and itself managed
    let a = es.ok(Method::GET, "/_alias/logs", None).await;
    assert_eq!(a["logs-000002"]["aliases"]["logs"]["is_write_index"], true, "{a}");
    assert_eq!(a["logs-000001"]["aliases"]["logs"]["is_write_index"], false);
    let s = es.ok(Method::GET, "/logs-000001/_settings", None).await;
    assert_eq!(s["logs-000001"]["settings"]["index"]["lifecycle"]["indexing_complete"], "true");
    // rollover alias missing → ERROR step with step_info, retryable
    es.ok(Method::PUT, "/noalias", Some(json!({"settings": {"index.lifecycle.name": "roll"}}))).await;
    let v = es.wait_for("/noalias/_ilm/explain", |_, v| v["indices"]["noalias"]["step"] == "ERROR").await;
    let e = &v["indices"]["noalias"];
    assert_eq!(e["failed_step"], "check-rollover-ready");
    assert_eq!(e["step_info"]["type"], "illegal_argument_exception");
    assert_eq!(e["step_info"]["reason"], "setting [index.lifecycle.rollover_alias] for index [noalias] is empty or not defined");
    assert_eq!(e["is_auto_retryable_error"], true);
    es.ok(Method::POST, "/noalias/_ilm/retry", None).await;
    // delete phase
    es.ok(Method::PUT, "/_ilm/policy/del", Some(json!({"policy": {"phases": {"delete": {"min_age": "0ms", "actions": {"delete": {}}}}}}))).await;
    es.ok(Method::PUT, "/short-lived", Some(json!({"settings": {"index.lifecycle.name": "del"}}))).await;
    es.wait_for("/short-lived", |st, _| st == 404).await;
    // history lands in the hidden ilm-history data stream
    let v = es.ok(Method::GET, "/_data_stream/ilm-history-7", None).await;
    assert_eq!(v["data_streams"][0]["hidden"], true);
    es.ok(Method::POST, "/ilm-history-7/_refresh", None).await;
    let v = es.ok(Method::POST, "/ilm-history-7/_search?expand_wildcards=all", Some(json!({"query": {"term": {"index": "short-lived"}}}))).await;
    assert!(v["hits"]["total"]["value"].as_u64().unwrap() >= 1, "{v}");
}

#[tokio::test]
async fn resize_apis() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.ok(Method::PUT, "/source", Some(json!({"settings": {"number_of_shards": 2, "number_of_replicas": 0, "index.merge.scheduler.max_merge_count": 4}}))).await;
    for i in 1..=3 {
        es.ok(Method::PUT, &format!("/source/_doc/{i}?refresh=true"), Some(json!({"foo": format!("hello world {i}")}))).await;
    }
    let (st, v) = es.post("/source/_clone/target", json!({"settings": {"index.number_of_shards": 6}})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["reason"], "the number of target shards (6) must be the same as the number of  source shards ( 2)");
    let (st, v) = es.post("/source/_clone/target", json!({})).await;
    assert_eq!(st, 500);
    assert_eq!(v["error"]["type"], "illegal_state_exception");
    assert_eq!(v["error"]["reason"], "index source must be read-only to resize index. use \"index.blocks.write=true\"");
    es.ok(Method::PUT, "/source/_settings", Some(json!({"index.blocks.write": true}))).await;
    let v = es.ok(Method::POST, "/source/_clone/target", Some(json!({"settings": {"index.merge.scheduler.max_thread_count": 2}, "aliases": {"t-alias": {}}}))).await;
    assert_eq!(v, json!({"acknowledged": true, "shards_acknowledged": true, "index": "target"}));
    let d = es.ok(Method::GET, "/target/_doc/2", None).await;
    assert_eq!(d["_index"], "target");
    assert_eq!(d["_source"]["foo"], "hello world 2");
    let s = &es.ok(Method::GET, "/target/_settings", None).await["target"]["settings"]["index"];
    assert_eq!(s["merge"]["scheduler"]["max_merge_count"], "4");
    assert_eq!(s["merge"]["scheduler"]["max_thread_count"], "2");
    assert_eq!(s["blocks"]["write"], "true");
    assert_eq!(s["number_of_shards"], "2");
    assert_eq!(s["resize"]["source"]["name"], "source");
    assert_eq!(es.ok(Method::GET, "/t-alias/_count", None).await["count"], 3);
    let (st, v) = es.post("/source/_clone/target", json!({})).await;
    assert_eq!((st, v["error"]["type"].as_str()), (400, Some("resource_already_exists_exception")));
    // shrink 2 → 1, split 2 → 4 (and illegal factors)
    es.ok(Method::POST, "/source/_shrink/shrunk", Some(json!({"settings": {"index.number_of_replicas": 0}}))).await;
    assert_eq!(es.ok(Method::GET, "/shrunk/_count", None).await["count"], 3);
    assert_eq!(es.ok(Method::GET, "/shrunk/_settings", None).await["shrunk"]["settings"]["index"]["number_of_shards"], "1");
    let (st, v) = es.post("/source/_shrink/x", json!({"settings": {"index.number_of_shards": 3}})).await;
    assert_eq!(st, 400, "{v}");
    let (st, v) = es.post("/shrunk/_shrink/x", json!({})).await;
    assert_eq!((st, v["error"]["reason"].as_str()), (400, Some("can't shrink an index with only one shard")));
    let (st, v) = es.post("/source/_split/x", json!({})).await;
    assert_eq!((st, v["error"]["type"].as_str()), (400, Some("action_request_validation_exception")));
    let (st, v) = es.post("/source/_split/x", json!({"settings": {"index.number_of_shards": 6}})).await;
    assert_eq!((st, v["error"]["type"].as_str()), (500, Some("illegal_state_exception")));
    es.ok(Method::POST, "/source/_split/split", Some(json!({"settings": {"index.number_of_shards": 4}}))).await;
    assert_eq!(es.ok(Method::GET, "/split/_count", None).await["count"], 3);
    // the copies are independent of the source
    es.ok(Method::PUT, "/split/_settings", Some(json!({"index.blocks.write": false}))).await;
    es.ok(Method::PUT, "/split/_doc/9?refresh=true", Some(json!({"foo": "new"}))).await;
    assert_eq!(es.ok(Method::GET, "/split/_count", None).await["count"], 4);
    assert_eq!(es.ok(Method::GET, "/source/_count", None).await["count"], 3);
    // freeze / unfreeze
    let (st, _) = es.call(Method::POST, "/source/_freeze", None).await;
    assert_eq!(st, 400);
    let resp = es.c.post(format!("{}/source/_unfreeze", es.base)).send().await.unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    assert!(resp.headers().get("warning").unwrap().to_str().unwrap().contains("Frozen indices are deprecated"));
}

#[tokio::test]
async fn data_stream_lifecycle_and_management() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    es.ok(
        Method::PUT,
        "/_index_template/t",
        Some(json!({"index_patterns": ["managed-*"], "data_stream": {}, "_meta": {"a": 1}, "template": {"settings": {"number_of_replicas": 0}, "lifecycle": {"data_retention": "30d"}}})),
    )
    .await;
    es.ok(Method::PUT, "/_data_stream/managed-a", None).await;
    let v = es.ok(Method::GET, "/_data_stream/managed-a", None).await;
    let ds = &v["data_streams"][0];
    assert_eq!(ds["lifecycle"]["data_retention"], "30d");
    assert_eq!(ds["lifecycle"]["enabled"], true);
    assert_eq!(ds["lifecycle"]["effective_retention"], "30d");
    assert_eq!(ds["lifecycle"]["retention_determined_by"], "data_stream_configuration");
    assert_eq!(ds["next_generation_managed_by"], "Data stream lifecycle");
    assert_eq!(ds["indices"][0]["managed_by"], "Data stream lifecycle");
    assert_eq!(ds["_meta"], json!({"a": 1}));
    let backing = ds["indices"][0]["index_name"].as_str().unwrap().to_string();
    let v = es.ok(Method::GET, &format!("/{backing}/_lifecycle/explain"), None).await;
    let e = &v["indices"][backing.as_str()];
    assert_eq!(e["managed_by_lifecycle"], true);
    assert_eq!(e["lifecycle"]["data_retention"], "30d");
    assert!(e["lifecycle"].get("rollover").is_none());
    let v = es.ok(Method::GET, "/managed-a/_lifecycle/explain?include_defaults=true", None).await;
    assert_eq!(v["indices"][backing.as_str()]["lifecycle"]["rollover"]["max_age"], "7d [automatic]");
    // get / put / delete lifecycle, global retention
    let v = es.ok(Method::GET, "/_data_stream/*/_lifecycle", None).await;
    assert_eq!(v["data_streams"][0]["lifecycle"]["data_retention"], "30d");
    assert_eq!(v["global_retention"], json!({}));
    let (st, v) = es.get("/_data_stream/managed-a,missing/_lifecycle").await;
    assert_eq!((st, v["error"]["reason"].as_str()), (404, Some("no such index [missing]")));
    es.ok(Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"data_streams.lifecycle.retention.max": "7d"}}))).await;
    let v = es.ok(Method::GET, "/_data_stream/managed-a/_lifecycle", None).await;
    assert_eq!(v["data_streams"][0]["lifecycle"]["effective_retention"], "7d");
    assert_eq!(v["data_streams"][0]["lifecycle"]["retention_determined_by"], "max_global_retention");
    assert_eq!(v["global_retention"]["max_retention"], "7d");
    es.ok(Method::PUT, "/_data_stream/managed-a/_lifecycle", Some(json!({"data_retention": "10d", "enabled": false}))).await;
    let v = es.ok(Method::GET, "/_data_stream/managed-a/_lifecycle", None).await;
    assert_eq!(v["data_streams"][0]["lifecycle"]["enabled"], false);
    es.ok(Method::DELETE, "/_data_stream/managed-a/_lifecycle", None).await;
    let v = es.ok(Method::GET, "/_data_stream/managed-a/_lifecycle", None).await;
    assert!(v["data_streams"][0].get("lifecycle").is_none());
    es.ok(Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"data_streams.lifecycle.retention.max": null}}))).await;
    // retention runs in the background
    es.ok(Method::PUT, "/_data_stream/managed-a/_lifecycle", Some(json!({"data_retention": "1s"}))).await;
    es.ok(Method::POST, "/managed-a/_doc?refresh=true&op_type=create", Some(json!({"@timestamp": 1593639273740i64}))).await;
    es.ok(Method::POST, "/managed-a/_rollover", None).await;
    es.ok(Method::PUT, "/_cluster/settings", Some(json!({"persistent": {"data_streams.lifecycle.poll_interval": "1s"}}))).await;
    es.wait_for("/_data_stream/managed-a", |_, v| v["data_streams"][0]["indices"].as_array().map(|a| a.len() == 1).unwrap_or(false)).await;
    let (st, _) = es.get(&format!("/{backing}")).await;
    assert_eq!(st, 404, "old generation deleted by retention");
    let v = es.ok(Method::GET, "/_lifecycle/stats", None).await;
    assert_eq!(v["data_stream_count"], 1);
    assert!(v["last_run_duration_in_millis"].is_number());

    // stats / modify / migrate / promote
    es.ok(Method::PUT, "/_index_template/plain", Some(json!({"index_patterns": ["plain-*", "migrate-me"], "priority": 5, "data_stream": {}}))).await;
    es.ok(Method::PUT, "/_data_stream/plain-a", None).await;
    es.ok(Method::POST, "/plain-a/_doc?refresh=true&op_type=create", Some(json!({"@timestamp": 1593639345064i64}))).await;
    let v = es.ok(Method::GET, "/_data_stream/_stats", None).await;
    assert_eq!(v["data_stream_count"], 2);
    let p = v["data_streams"].as_array().unwrap().iter().find(|d| d["data_stream"] == "plain-a").unwrap().clone();
    assert_eq!(p["maximum_timestamp"], 1593639345064i64);
    es.ok(Method::PUT, "/standalone", Some(json!({"mappings": {"properties": {"@timestamp": {"type": "date"}}}}))).await;
    es.ok(Method::POST, "/_data_stream/_modify", Some(json!({"actions": [{"add_backing_index": {"data_stream": "plain-a", "index": "standalone"}}]}))).await;
    let v = es.ok(Method::GET, "/_data_stream/plain-a", None).await;
    assert_eq!(v["data_streams"][0]["indices"][0]["index_name"], "standalone");
    assert_eq!(v["data_streams"][0]["generation"], 2);
    let write = v["data_streams"][0]["indices"][1]["index_name"].as_str().unwrap().to_string();
    let (st, v) = es.post("/_data_stream/_modify", json!({"actions": [{"remove_backing_index": {"data_stream": "plain-a", "index": write}}]})).await;
    assert_eq!(st, 400);
    assert!(v["error"]["reason"].as_str().unwrap().contains("because it is the write index"));
    let (st, v) = es.delete(&format!("/{write}")).await;
    assert_eq!(st, 400, "{v}");
    es.ok(Method::DELETE, "/standalone", None).await;
    assert_eq!(es.ok(Method::GET, "/_data_stream/plain-a", None).await["data_streams"][0]["generation"], 3);
    es.ok(Method::PUT, "/m1", Some(json!({"aliases": {"migrate-me": {"is_write_index": true}}}))).await;
    es.ok(Method::POST, "/m1/_doc?refresh=true", Some(json!({"@timestamp": "2009-11-15T14:12:12"}))).await;
    es.ok(Method::PUT, "/m0", Some(json!({"aliases": {"migrate-me": {}}}))).await;
    es.ok(Method::POST, "/m0/_doc?refresh=true", Some(json!({"@timestamp": "2009-11-15T14:12:13"}))).await;
    es.ok(Method::POST, "/_data_stream/_migrate/migrate-me", None).await;
    let v = es.ok(Method::GET, "/_data_stream/migrate-me", None).await;
    let names: Vec<&str> = v["data_streams"][0]["indices"].as_array().unwrap().iter().map(|i| i["index_name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["m0", "m1"]);
    assert_eq!(es.ok(Method::GET, "/m0/_settings", None).await["m0"]["settings"]["index"]["hidden"], "true");
    // ES registers `POST /_data_stream/_promote/{name}`.
    let v = es.ok(Method::POST, "/_data_stream/_promote/migrate-me", None).await;
    assert_eq!(v, json!({"acknowledged": true}));
    let (st, v) = es.call(Method::POST, "/_data_stream/_promote/nope", None).await;
    assert_eq!((st, v["error"]["reason"].clone()), (404, json!("data stream [nope] does not exist")));
}

#[tokio::test]
async fn slm_policies() {
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;
    let (st, _) = es.get("/_slm/policy/daily").await;
    assert_eq!(st, 404);
    let body = json!({"schedule": "0 1 2 3 4 ?", "name": "<prod-snap-{now/d}>", "repository": "repo", "config": {"indices": ["foo-*"]}, "retention": {"expire_after": "30d", "min_count": 1, "max_count": 50}});
    // ES checks that the repository exists before the schedule interval.
    let (st, v) = es.put("/_slm/policy/daily", json!({"schedule": "0 * * * * ?", "name": "x", "repository": "repo"})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["reason"], "no such repository [repo]");
    let loc = tmp.path().join("slm-repo");
    es.ok(Method::PUT, "/_snapshot/repo", Some(json!({"type": "fs", "settings": {"location": loc.to_str().unwrap()}}))).await;
    let (st, v) = es.put("/_slm/policy/daily", json!({"schedule": "0 * * * * ?", "name": "x", "repository": "repo"})).await;
    assert_eq!(st, 400);
    assert_eq!(v["error"]["reason"], "invalid schedule [0 * * * * ?]: schedule would be too frequent, executing more than every [15m]");
    es.ok(Method::PUT, "/_slm/policy/daily", Some(body.clone())).await;
    let v = es.ok(Method::GET, "/_slm/policy/daily", None).await;
    assert_eq!(v["daily"]["version"], 1);
    assert!(v["daily"]["next_execution_millis"].as_i64().unwrap() > 0);
    assert_eq!(v["daily"]["stats"]["policy"], "daily");
    let modified = v["daily"]["modified_date_millis"].clone();
    es.ok(Method::PUT, "/_slm/policy/daily", Some(body)).await;
    let v = es.ok(Method::GET, "/_slm/policy/daily", None).await;
    assert_eq!((v["daily"]["version"].clone(), v["daily"]["modified_date_millis"].clone()), (json!(1), modified));
    // Unregister the repository so the execution below fails.
    es.ok(Method::DELETE, "/_snapshot/repo", None).await;
    let v = es.ok(Method::PUT, "/_slm/policy/daily/_execute", None).await;
    assert!(v["snapshot_name"].as_str().unwrap().starts_with("prod-snap-"));
    let v = es.ok(Method::GET, "/_slm/policy/daily", None).await;
    assert!(v["daily"]["last_failure"]["details"].as_str().unwrap().contains("repository_missing_exception"));
    assert_eq!(v["daily"]["stats"]["snapshots_failed"], 1);
    let v = es.ok(Method::GET, "/_slm/stats", None).await;
    assert_eq!(v["total_snapshots_failed"], 1);
    assert_eq!(es.ok(Method::GET, "/_slm/status", None).await["operation_mode"], "RUNNING");
    es.ok(Method::POST, "/_slm/stop", None).await;
    // Like ES, stop reports STOPPING until the scheduler settles it to STOPPED.
    let mode = es.ok(Method::GET, "/_slm/status", None).await["operation_mode"].clone();
    assert!(mode == "STOPPING" || mode == "STOPPED", "{mode}");
    es.ok(Method::POST, "/_slm/start", None).await;
    // an ILM wait_for_snapshot may now reference the policy
    es.ok(Method::PUT, "/_ilm/policy/w", Some(json!({"policy": {"phases": {"delete": {"actions": {"wait_for_snapshot": {"policy": "daily"}, "delete": {}}}}}}))).await;
    es.ok(Method::DELETE, "/_slm/policy/daily", None).await;
    let (st, _) = es.delete("/_slm/policy/daily").await;
    assert_eq!(st, 404);
}
