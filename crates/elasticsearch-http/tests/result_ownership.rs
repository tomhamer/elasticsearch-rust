//! Security contexts of retained results and background work, over a
//! security-enabled node:
//!
//! * async search results belong to the subject that submitted them (user +
//!   realm, or API key; run-as aware): another subject that knows the id
//!   finds nothing, `monitor` reads any status and `cancel_task` deletes any
//!   search (ES `AsyncTaskIndexService`); ES|QL / SQL / EQL async results
//!   follow the same rules and SQL cursors are their creator's;
//! * enrich policies need `read` on their sources to be stored and executed,
//!   keep the storing subject and execute with its document / field level
//!   security;
//! * scheduled watches run as the subject that stored them, with its DLS /
//!   FLS, whoever made the latest watcher request; `_execute` runs as the
//!   caller (ES).

use base64::Engine;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::security::{hash::Hasher, SecConfig};
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

const PW: &str = "ownership-test-password";

struct Es {
    base: String,
    client: Client,
    _task: tokio::task::JoinHandle<()>,
}

async fn start(dir: &Path) -> (Es, Arc<Node>) {
    let node = Node::open(dir).unwrap();
    node.node_settings.write().insert("xpack.license.self_generated.type".into(), json!("trial"));
    let mut cfg = SecConfig::default();
    cfg.enabled = true;
    cfg.api_key_enabled = true;
    cfg.trial = true;
    cfg.file_users.insert("admin".into(), Hasher::Pbkdf2(1000).hash(PW));
    cfg.file_user_roles.insert("admin".into(), vec!["superuser".into()]);
    node.security.configure(cfg);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let n = node.clone();
    let task = tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(n, listener).await;
    });
    (Es { base: format!("http://{addr}"), client: Client::builder().timeout(Duration::from_secs(30)).build().unwrap(), _task: task }, node)
}

/// How a request authenticates.
#[derive(Clone)]
enum As {
    User(&'static str),
    Key(String),
    /// `admin` running as this user (`es-security-runas-user`).
    RunAs(&'static str),
}

const ADMIN: As = As::User("admin");
const ALICE: As = As::User("alice");
const BOB: As = As::User("bob");
const CAROL: As = As::User("carol");
const DAVE: As = As::User("dave");

impl Es {
    async fn call(&self, who: &As, m: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        let basic = |u: &str| format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{u}:{PW}")));
        let mut r = self.client.request(m, format!("{}{}", self.base, path));
        r = match who {
            As::User(u) => r.header("authorization", basic(u)),
            As::Key(k) => r.header("authorization", format!("ApiKey {k}")),
            As::RunAs(u) => r.header("authorization", basic("admin")).header("es-security-runas-user", *u),
        };
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b.to_string());
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let text = resp.text().await.unwrap();
        (st, if text.is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap_or(Value::String(text)) })
    }
    async fn get(&self, w: &As, p: &str) -> (u16, Value) {
        self.call(w, Method::GET, p, None).await
    }
    async fn post(&self, w: &As, p: &str, b: Value) -> (u16, Value) {
        self.call(w, Method::POST, p, Some(b)).await
    }
    async fn post0(&self, w: &As, p: &str) -> (u16, Value) {
        self.call(w, Method::POST, p, None).await
    }
    async fn put(&self, w: &As, p: &str, b: Value) -> (u16, Value) {
        self.call(w, Method::PUT, p, Some(b)).await
    }
    async fn delete(&self, w: &As, p: &str) -> (u16, Value) {
        self.call(w, Method::DELETE, p, None).await
    }
    async fn bulk(&self, lines: &[Value]) {
        let body: String = lines.iter().map(|l| format!("{l}\n")).collect();
        let resp = self
            .client
            .post(format!("{}/_bulk?refresh=true", self.base))
            .header("authorization", format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("admin:{PW}"))))
            .header("content-type", "application/x-ndjson")
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        let v: Value = resp.json().await.unwrap();
        assert_eq!(v["errors"], false, "{v}");
    }
    async fn api_key(&self, owner: &As, name: &str) -> As {
        let (st, r) = self.post(owner, "/_security/api_key", json!({"name": name})).await;
        assert_eq!(st, 200, "{r}");
        As::Key(r["encoded"].as_str().unwrap().to_string())
    }
}

/// A 404 that reads as "no such result".
fn assert_missing(st: u16, v: &Value, what: &str) {
    assert_eq!(st, 404, "{what}: {v}");
}

async fn setup(es: &Es) {
    let (st, r) = es
        .put(&ADMIN, "/docs", json!({"mappings": {"properties": {"name": {"type": "keyword"}, "dept": {"type": "keyword"}, "email": {"type": "keyword"}, "salary": {"type": "long"},
            "obj": {"properties": {"public": {"type": "keyword"}, "secret": {"type": "keyword"}}}}}}))
        .await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = es
        .put(&ADMIN, "/ev", json!({"mappings": {"properties": {"@timestamp": {"type": "date"}, "event": {"properties": {"category": {"type": "keyword"}}}, "user": {"type": "keyword"}}}}))
        .await;
    assert_eq!(st, 200, "{r}");
    let mut lines = Vec::new();
    for (id, name, dept, salary) in [("1", "A", "eng", 100), ("2", "B", "eng", 200), ("3", "C", "sales", 300)] {
        lines.push(json!({"index": {"_index": "docs", "_id": id}}));
        lines.push(json!({"name": name, "dept": dept, "email": format!("{}@x", name.to_lowercase()), "salary": salary, "obj": {"public": format!("p{id}"), "secret": format!("s{id}")}}));
    }
    for (id, ts, cat) in [("1", "2024-01-01T00:00:01Z", "process"), ("2", "2024-01-01T00:00:02Z", "file")] {
        lines.push(json!({"index": {"_index": "ev", "_id": id}}));
        lines.push(json!({"@timestamp": ts, "event": {"category": cat}, "user": "u"}));
    }
    es.bulk(&lines).await;
    let read = json!([{"names": ["docs", "ev"], "privileges": ["read", "view_index_metadata"]}]);
    let roles = [
        ("reader", json!({"cluster": ["manage_own_api_key"], "indices": read})),
        ("monitor_reader", json!({"cluster": ["monitor"], "indices": read})),
        ("canceller", json!({"cluster": ["cancel_task"], "indices": read})),
        // Enrich administration without read on the source.
        ("enrich_noread", json!({"cluster": ["manage_enrich"], "indices": [{"names": ["ev"], "privileges": ["read"]}]})),
        // Enrich administration reading the source under DLS + FLS.
        ("enrich_restricted", json!({"cluster": ["manage_enrich"], "indices": [{"names": ["docs"], "privileges": ["read", "view_index_metadata"],
            "query": {"term": {"dept": "eng"}}, "field_security": {"grant": ["*"], "except": ["salary", "obj.secret"]}}]})),
        // A watch owner reading the source under DLS + FLS.
        ("watch_restricted", json!({"cluster": ["manage_watcher"], "indices": [{"names": ["docs"], "privileges": ["read"],
            "query": {"term": {"dept": "eng"}}, "field_security": {"grant": ["*"], "except": ["salary"]}}]})),
    ];
    for (name, body) in roles {
        let (st, r) = es.put(&ADMIN, &format!("/_security/role/{name}"), body).await;
        assert_eq!(st, 200, "role {name}: {r}");
    }
    for (user, roles) in [
        ("alice", json!(["reader"])),
        ("bob", json!(["reader"])),
        ("carol", json!(["monitor_reader"])),
        ("dave", json!(["canceller"])),
        ("ena", json!(["enrich_noread"])),
        ("enb", json!(["enrich_restricted"])),
        ("wat", json!(["watch_restricted"])),
    ] {
        let (st, r) = es.put(&ADMIN, &format!("/_security/user/{user}"), json!({"password": PW, "roles": roles})).await;
        assert_eq!(st, 200, "user {user}: {r}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn async_search_is_bound_to_its_submitter() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    setup(&es).await;

    let (st, v) = es.post(&ALICE, "/docs/_async_search?keep_on_completion=true&wait_for_completion_timeout=10s&keep_alive=1h", json!({"query": {"match_all": {}}})).await;
    assert_eq!(st, 200, "{v}");
    let id = v["id"].as_str().unwrap().to_string();
    let (st, mine) = es.get(&ALICE, &format!("/_async_search/{id}")).await;
    assert_eq!(st, 200, "{mine}");
    assert_eq!(mine["response"]["hits"]["total"]["value"], 3);
    let expiration = mine["expiration_time_in_millis"].clone();

    // Another user with the same privileges knows the id: nothing there.
    let (st, v) = es.get(&BOB, &format!("/_async_search/{id}")).await;
    assert_missing(st, &v, "bob get");
    assert_eq!(v["error"]["type"], "resource_not_found_exception", "{v}");
    // ... cannot extend it ...
    let (st, v) = es.get(&BOB, &format!("/_async_search/{id}?keep_alive=10d")).await;
    assert_missing(st, &v, "bob extend");
    let (_, again) = es.get(&ALICE, &format!("/_async_search/{id}")).await;
    assert_eq!(again["expiration_time_in_millis"], expiration, "a non-owner's keep_alive must not extend the search");
    // ... nor read its status (no `monitor`) ...
    let (st, v) = es.get(&BOB, &format!("/_async_search/status/{id}")).await;
    assert_missing(st, &v, "bob status");
    // ... nor delete it (no `cancel_task`).
    let (st, v) = es.delete(&BOB, &format!("/_async_search/{id}")).await;
    assert_missing(st, &v, "bob delete");

    // The submitter reads its own status without `monitor`; `monitor` reads any status.
    let (st, v) = es.get(&ALICE, &format!("/_async_search/status/{id}")).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.get(&CAROL, &format!("/_async_search/status/{id}")).await;
    assert_eq!(st, 200, "{v}");
    // `monitor` does not read the results.
    let (st, v) = es.get(&CAROL, &format!("/_async_search/{id}")).await;
    assert_missing(st, &v, "carol get");
    // Nor does a superuser that did not submit it (ES has no superuser exception for reads).
    let (st, v) = es.get(&ADMIN, &format!("/_async_search/{id}")).await;
    assert_missing(st, &v, "admin get");
    // The submitter's API key is a different subject.
    let alice_key = es.api_key(&ALICE, "alice-k1").await;
    let (st, v) = es.get(&alice_key, &format!("/_async_search/{id}")).await;
    assert_missing(st, &v, "alice's key get");
    // Run-as: the effective user owns.
    let (st, v) = es.get(&As::RunAs("alice"), &format!("/_async_search/{id}")).await;
    assert_eq!(st, 200, "admin run as alice: {v}");
    let (st, v) = es.get(&As::RunAs("bob"), &format!("/_async_search/{id}")).await;
    assert_missing(st, &v, "admin run as bob");

    // `cancel_task` deletes any search.
    let (st, v) = es.delete(&DAVE, &format!("/_async_search/{id}")).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.get(&ALICE, &format!("/_async_search/{id}")).await;
    assert_missing(st, &v, "deleted");

    // API keys own their own searches: another key of the same user does not.
    let k1 = es.api_key(&ALICE, "k1").await;
    let k2 = es.api_key(&ALICE, "k2").await;
    let (st, v) = es.post(&k1, "/docs/_async_search?keep_on_completion=true&wait_for_completion_timeout=10s", json!({})).await;
    assert_eq!(st, 200, "{v}");
    let kid = v["id"].as_str().unwrap().to_string();
    let (st, v) = es.get(&k1, &format!("/_async_search/{kid}")).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.get(&k2, &format!("/_async_search/{kid}")).await;
    assert_missing(st, &v, "other key get");
    let (st, v) = es.get(&ALICE, &format!("/_async_search/{kid}")).await;
    assert_missing(st, &v, "key owner get");
    let (st, v) = es.delete(&k2, &format!("/_async_search/{kid}")).await;
    assert_missing(st, &v, "other key delete");
    let (st, v) = es.delete(&k1, &format!("/_async_search/{kid}")).await;
    assert_eq!(st, 200, "{v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn query_language_results_and_cursors_are_bound_to_their_creator() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    setup(&es).await;

    // ES|QL async.
    let (st, v) = es.post(&ALICE, "/_query/async", json!({"query": "FROM docs | STATS c = COUNT(*)", "keep_on_completion": true})).await;
    assert_eq!(st, 200, "{v}");
    let id = v["id"].as_str().unwrap().to_string();
    let (st, v) = es.get(&BOB, &format!("/_query/async/{id}")).await;
    assert_missing(st, &v, "esql bob get");
    let (st, v) = es.post0(&BOB, &format!("/_query/async/{id}/stop")).await;
    assert_missing(st, &v, "esql bob stop");
    let (st, v) = es.delete(&BOB, &format!("/_query/async/{id}")).await;
    assert_missing(st, &v, "esql bob delete");
    let (st, v) = es.get(&ALICE, &format!("/_query/async/{id}")).await;
    assert_eq!(st, 200, "{v}");
    assert!(v.get("__owner").is_none(), "{v}");
    assert_eq!(v["values"], json!([[3]]), "{v}");
    let (st, v) = es.delete(&DAVE, &format!("/_query/async/{id}")).await;
    assert_eq!(st, 200, "cancel_task deletes: {v}");
    let (st, v) = es.get(&ALICE, &format!("/_query/async/{id}")).await;
    assert_missing(st, &v, "esql deleted");

    // SQL cursor: another user cannot page or close it; the owner still can.
    let (st, v) = es.post(&ALICE, "/_sql", json!({"query": "SELECT name FROM docs ORDER BY name", "fetch_size": 1})).await;
    assert_eq!(st, 200, "{v}");
    let cursor = v["cursor"].as_str().unwrap().to_string();
    let (st, v) = es.post(&BOB, "/_sql", json!({"cursor": cursor})).await;
    assert_eq!(st, 404, "bob page: {v}");
    assert_eq!(v["error"]["caused_by"]["type"], "search_context_missing_exception", "{v}");
    let (st, v) = es.post(&BOB, "/_sql/close", json!({"cursor": cursor})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["succeeded"], false, "bob close: {v}");
    let (st, v) = es.post(&ALICE, "/_sql", json!({"cursor": cursor})).await;
    assert_eq!(st, 200, "alice page: {v}");
    assert_eq!(v["rows"], json!([["B"]]), "{v}");

    // SQL async results.
    let (st, v) = es.post(&ALICE, "/_sql", json!({"query": "SELECT COUNT(*) FROM docs", "wait_for_completion_timeout": "10s", "keep_on_completion": true})).await;
    assert_eq!(st, 200, "{v}");
    let sid = v["id"].as_str().unwrap().to_string();
    let (st, v) = es.get(&BOB, &format!("/_sql/async/{sid}")).await;
    assert_missing(st, &v, "sql bob get");
    let (st, v) = es.get(&BOB, &format!("/_sql/async/status/{sid}")).await;
    assert_missing(st, &v, "sql bob status");
    let (st, v) = es.get(&CAROL, &format!("/_sql/async/status/{sid}")).await;
    assert_eq!(st, 200, "monitor reads sql status: {v}");
    let (st, v) = es.get(&CAROL, &format!("/_sql/async/{sid}")).await;
    assert_missing(st, &v, "monitor does not read sql results");
    let (st, v) = es.delete(&BOB, &format!("/_sql/async/delete/{sid}")).await;
    assert_missing(st, &v, "sql bob delete");
    let (st, v) = es.get(&ALICE, &format!("/_sql/async/{sid}")).await;
    assert_eq!(st, 200, "{v}");
    assert!(v.get("__owner").is_none(), "{v}");
    let (st, v) = es.delete(&ALICE, &format!("/_sql/async/delete/{sid}")).await;
    assert_eq!(st, 200, "{v}");

    // EQL async results.
    let (st, v) = es.post(&ALICE, "/ev/_eql/search?wait_for_completion_timeout=10s&keep_on_completion=true", json!({"query": "any where true"})).await;
    assert_eq!(st, 200, "{v}");
    let eid = v["id"].as_str().unwrap().to_string();
    let (st, v) = es.get(&BOB, &format!("/_eql/search/{eid}")).await;
    assert_missing(st, &v, "eql bob get");
    let (st, v) = es.get(&BOB, &format!("/_eql/search/status/{eid}")).await;
    assert_missing(st, &v, "eql bob status");
    let (st, v) = es.get(&CAROL, &format!("/_eql/search/status/{eid}")).await;
    assert_eq!(st, 200, "monitor reads eql status: {v}");
    let (st, v) = es.delete(&BOB, &format!("/_eql/search/{eid}")).await;
    assert_missing(st, &v, "eql bob delete");
    let (st, v) = es.get(&ALICE, &format!("/_eql/search/{eid}")).await;
    assert_eq!(st, 200, "{v}");
    assert!(v.get("__owner").is_none(), "{v}");
    assert_eq!(v["hits"]["events"].as_array().map(Vec::len), Some(2), "{v}");
    let (st, v) = es.delete(&DAVE, &format!("/_eql/search/{eid}")).await;
    assert_eq!(st, 200, "cancel_task deletes eql results: {v}");
}

/// The `_source`s of the enrich index of `policy`.
async fn enrich_docs(es: &Es, policy: &str) -> Vec<Value> {
    let (st, v) = es.post(&ADMIN, &format!("/.enrich-{policy}/_search?expand_wildcards=all&size=100"), json!({"query": {"match_all": {}}})).await;
    assert_eq!(st, 200, "{v}");
    v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_source"].clone()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn enrich_reads_sources_as_the_policy_owner() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    setup(&es).await;
    let ena = As::User("ena");
    let enb = As::User("enb");
    let policy = json!({"match": {"indices": "docs", "match_field": "email", "enrich_fields": ["name", "dept", "salary"]}});

    // An enrich administrator without read on the source cannot store a policy over it.
    let (st, v) = es.put(&ena, "/_enrich/policy/p_noread", policy.clone()).await;
    assert_eq!(st, 403, "{v}");
    assert_eq!(v["error"]["type"], "security_exception", "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().contains("unable to store policy because no indices match with the specified index patterns [docs]"), "{v}");
    let (_, v) = es.get(&ADMIN, "/_enrich/policy/p_noread").await;
    assert_eq!(v["policies"], json!([]), "{v}");
    // ... nor execute one stored by someone who can read it.
    let (st, v) = es.put(&ADMIN, "/_enrich/policy/p_admin", policy.clone()).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post0(&ena, "/_enrich/policy/p_admin/_execute").await;
    assert_eq!(st, 403, "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().contains("on indices [docs]"), "{v}");
    // The stored headers never show in the policy's configuration.
    let (_, v) = es.get(&ADMIN, "/_enrich/policy/p_admin").await;
    assert_eq!(v["policies"][0]["config"], json!({"match": {"name": "p_admin", "indices": ["docs"], "match_field": "email", "enrich_fields": ["name", "dept", "salary"]}}), "{v}");

    // A source reader under DLS + FLS stores and executes a policy: the
    // enrich index holds what that reader may see, nothing more.
    let (st, v) = es.put(&enb, "/_enrich/policy/p_restricted", policy.clone()).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post0(&enb, "/_enrich/policy/p_restricted/_execute").await;
    assert_eq!(st, 200, "{v}");
    let check = |docs: &[Value]| {
        let mut emails: Vec<&str> = docs.iter().map(|d| d["email"].as_str().unwrap()).collect();
        emails.sort();
        assert_eq!(emails, ["a@x", "b@x"], "only the reader's documents: {docs:?}");
        for d in docs {
            assert!(d.get("salary").is_none(), "a field the reader cannot see: {d}");
            assert!(d.get("name").is_some() && d.get("dept").is_some(), "{d}");
        }
    };
    check(&enrich_docs(&es, "p_restricted").await);
    // Executed by a superuser, the policy still reads as the subject that stored it.
    let (st, v) = es.post0(&ADMIN, "/_enrich/policy/p_restricted/_execute").await;
    assert_eq!(st, 200, "{v}");
    check(&enrich_docs(&es, "p_restricted").await);
    // The administrator's own policy reads everything.
    let (st, v) = es.post0(&ADMIN, "/_enrich/policy/p_admin/_execute").await;
    assert_eq!(st, 200, "{v}");
    let all = enrich_docs(&es, "p_admin").await;
    assert_eq!(all.len(), 3, "{all:?}");
    assert!(all.iter().all(|d| d.get("salary").is_some()), "{all:?}");

    // An object field partly hidden by FLS is never copied whole.
    let (st, v) = es.put(&enb, "/_enrich/policy/p_object", json!({"match": {"indices": "docs", "match_field": "email", "enrich_fields": ["obj"]}})).await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post0(&enb, "/_enrich/policy/p_object/_execute").await;
    if st == 200 {
        for d in enrich_docs(&es, "p_object").await {
            assert!(d.pointer("/obj/secret").is_none(), "{d}");
        }
    } else {
        assert_eq!(st, 403, "{v}");
    }

    // Once the stored subject loses read on the source, executions stop.
    let (st, v) = es.delete(&ADMIN, "/_security/role/enrich_restricted").await;
    assert_eq!(st, 200, "{v}");
    let (st, v) = es.post0(&ADMIN, "/_enrich/policy/p_restricted/_execute").await;
    assert_eq!(st, 403, "{v}");
}

/// History records of `watch` (oldest first) once there are at least `n`.
async fn history(es: &Es, watch: &str, n: usize) -> Vec<Value> {
    for _ in 0..120 {
        let (st, v) = es.post(&ADMIN, "/.watcher-history-18*/_search?size=100&sort=@timestamp:asc", json!({"query": {"term": {"watch_id": watch}}})).await;
        if st == 200 {
            let hits: Vec<Value> = v["hits"]["hits"].as_array().cloned().unwrap_or_default().into_iter().map(|h| h["_source"].clone()).collect();
            if hits.len() >= n {
                return hits;
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("no {n} history records for watch [{watch}]");
}

/// The search input's hits of a watch record: `(names, any salary shown)`.
fn input_hits(rec: &Value) -> (Vec<String>, bool) {
    let hits = rec["result"]["input"]["payload"]["hits"]["hits"].as_array().cloned().unwrap_or_default();
    let mut names: Vec<String> = hits.iter().map(|h| h["_source"]["name"].as_str().unwrap_or("?").to_string()).collect();
    names.sort();
    (names, hits.iter().any(|h| h["_source"].get("salary").is_some()))
}

#[tokio::test(flavor = "multi_thread")]
async fn watches_run_as_the_subject_that_stored_them() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    setup(&es).await;
    let wat = As::User("wat");
    let watch = json!({
        "trigger": {"schedule": {"interval": "1s"}},
        "input": {"search": {"request": {"indices": ["docs"], "body": {"query": {"match_all": {}}, "size": 10}}}},
        "condition": {"always": {}},
        "actions": {"log": {"logging": {"text": "hits {{ctx.payload.hits.total}}"}}}
    });
    let (st, v) = es.put(&wat, "/_watcher/watch/w_restricted", watch.clone()).await;
    assert_eq!(st, 201, "{v}");
    let (st, v) = es.put(&ADMIN, "/_watcher/watch/w_admin", watch.clone()).await;
    assert_eq!(st, 201, "{v}");
    // Two principals alternate watcher requests while the watches fire.
    for _ in 0..6 {
        assert_eq!(es.get(&ADMIN, "/_watcher/watch/w_admin").await.0, 200);
        assert_eq!(es.get(&wat, "/_watcher/watch/w_restricted").await.0, 200);
        assert_eq!(es.get(&ADMIN, "/_watcher/stats").await.0, 200);
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    for rec in history(&es, "w_restricted", 3).await {
        assert_eq!(rec["user"], "wat", "{rec}");
        assert_eq!(rec["result"]["input"]["status"], "success", "{rec}");
        let (names, salary) = input_hits(&rec);
        assert_eq!(names, ["A", "B"], "the owner's DLS: {rec}");
        assert!(!salary, "the owner's FLS: {rec}");
    }
    for rec in history(&es, "w_admin", 3).await {
        assert_eq!(rec["user"], "admin", "{rec}");
        let (names, salary) = input_hits(&rec);
        assert_eq!(names, ["A", "B", "C"], "{rec}");
        assert!(salary, "{rec}");
    }
    // The stored headers never show.
    let (_, v) = es.get(&ADMIN, "/_watcher/watch/w_restricted").await;
    assert!(v["status"].get("headers").is_none(), "{v}");

    // `_execute` of a stored watch runs as the caller (ES
    // `TransportExecuteWatchAction` swaps in the request's security
    // headers), run-as included; scheduled runs stay the owner's.
    let (st, v) = es.post(&wat, "/_watcher/watch/w_restricted/_execute", json!({})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["watch_record"]["user"], "wat", "{v}");
    let (names, salary) = input_hits(&v["watch_record"]);
    assert_eq!(names, ["A", "B"], "{v}");
    assert!(!salary, "{v}");
    let (st, v) = es.post(&ADMIN, "/_watcher/watch/w_restricted/_execute", json!({})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["watch_record"]["user"], "admin", "{v}");
    assert_eq!(input_hits(&v["watch_record"]).0, ["A", "B", "C"], "{v}");
    let (st, v) = es.post(&As::RunAs("wat"), "/_watcher/watch/w_admin/_execute", json!({})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["watch_record"]["user"], "wat", "{v}");
    assert_eq!(input_hits(&v["watch_record"]).0, ["A", "B"], "{v}");
    // An inline watch runs as the caller.
    let (st, v) = es.post(&wat, "/_watcher/watch/_execute", json!({"watch": watch})).await;
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["watch_record"]["user"], "wat", "{v}");
    let (names, salary) = input_hits(&v["watch_record"]);
    assert_eq!(names, ["A", "B"], "{v}");
    assert!(!salary, "{v}");

    // An update by another subject rebinds the watch to that subject (ES
    // stores the updating request's headers): its scheduled runs follow.
    let (st, v) = es.put(&ADMIN, "/_watcher/watch/w_restricted", watch.clone()).await;
    assert_eq!(st, 200, "{v}");
    let before = history(&es, "w_restricted", 1).await.len();
    let recs = history(&es, "w_restricted", before + 2).await;
    assert_eq!(recs.last().unwrap()["user"], "admin", "{:?}", recs.last());
    for id in ["w_restricted", "w_admin"] {
        es.delete(&ADMIN, &format!("/_watcher/watch/{id}")).await;
    }
}
