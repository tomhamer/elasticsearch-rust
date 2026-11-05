//! Document- and field-level security end to end: roles with `query` (DLS) and
//! `field_security` (FLS), native users, API keys, over a security-enabled node.

use base64::Engine;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::security::{hash::Hasher, SecConfig};
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;

const PW: &str = "dls-fls-test-password";

struct Es {
    base: String,
    client: Client,
    _task: tokio::task::JoinHandle<()>,
}

async fn start(dir: &Path) -> (Es, Arc<Node>) {
    let node = Node::open(dir).unwrap();
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
    (Es { base: format!("http://{addr}"), client: Client::builder().timeout(std::time::Duration::from_secs(20)).build().unwrap(), _task: task }, node)
}

/// How a request authenticates.
#[derive(Clone)]
enum As {
    User(&'static str),
    Key(String),
}

fn auth(a: &As) -> String {
    match a {
        As::User(u) => format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{u}:{PW}"))),
        As::Key(k) => format!("ApiKey {k}"),
    }
}

impl Es {
    async fn call(&self, who: &As, m: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.client.request(m, format!("{}{}", self.base, path)).header("authorization", auth(who));
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b.to_string());
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let text = resp.text().await.unwrap();
        (st, if text.is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap_or(Value::String(text)) })
    }
    async fn ndjson(&self, who: &As, path: &str, lines: &[Value]) -> (u16, Value) {
        let body: String = lines.iter().map(|l| format!("{l}\n")).collect();
        let resp = self.client.post(format!("{}{}", self.base, path)).header("authorization", auth(who)).header("content-type", "application/x-ndjson").body(body).send().await.unwrap();
        let st = resp.status().as_u16();
        (st, resp.json().await.unwrap_or(Value::Null))
    }
    async fn get(&self, w: &As, p: &str) -> (u16, Value) {
        self.call(w, Method::GET, p, None).await
    }
    async fn post(&self, w: &As, p: &str, b: Value) -> (u16, Value) {
        self.call(w, Method::POST, p, Some(b)).await
    }
    async fn put(&self, w: &As, p: &str, b: Value) -> (u16, Value) {
        self.call(w, Method::PUT, p, Some(b)).await
    }
}

fn ids(v: &Value) -> Vec<String> {
    let mut out: Vec<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
    out.sort();
    out
}

const ADMIN: As = As::User("admin");

async fn setup(es: &Es) {
    let (st, r) = es
        .put(
            &ADMIN,
            "/docs",
            json!({"mappings": {"properties": {
                "name": {"type": "keyword"},
                "dept": {"type": "keyword"},
                "owner": {"type": "keyword"},
                "salary": {"type": "long"},
                "private": {"type": "text"},
                "note": {"type": "text"},
                "obj": {"properties": {"public": {"type": "keyword"}, "secret": {"type": "keyword"}}}
            }}}),
        )
        .await;
    assert_eq!(st, 200, "{r}");
    let docs = [
        ("1", json!({"name": "A", "dept": "eng", "owner": "alice", "salary": 100, "private": "needle one", "note": "shared haystack", "obj": {"public": "p1", "secret": "s1"}})),
        ("2", json!({"name": "B", "dept": "eng", "owner": "bob", "salary": 200, "private": "needle two", "note": "shared haystack", "obj": {"public": "p2", "secret": "s2"}})),
        ("3", json!({"name": "C", "dept": "sales", "owner": "carol", "salary": 300, "private": "needle three", "note": "shared haystack", "obj": {"public": "p3", "secret": "s3"}})),
    ];
    let mut lines = Vec::new();
    for (id, d) in docs {
        lines.push(json!({"index": {"_index": "docs", "_id": id}}));
        lines.push(d);
    }
    let (st, r) = es.ndjson(&ADMIN, "/_bulk?refresh=true", &lines).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["errors"], false, "{r}");

    let roles = [
        (
            "eng_nosalary",
            json!({"cluster": ["manage_own_api_key"], "indices": [{"names": ["docs"], "privileges": ["read", "view_index_metadata"], "query": {"term": {"dept": "eng"}},
                "field_security": {"grant": ["*"], "except": ["salary", "private", "obj.secret"]}}]}),
        ),
        ("sales_only", json!({"indices": [{"names": ["docs"], "privileges": ["read"], "query": "{\"term\":{\"dept\":\"sales\"}}"}]})),
        (
            "mine",
            json!({"indices": [{"names": ["docs"], "privileges": ["read"],
                "query": {"template": {"source": "{\"term\":{\"owner\":\"{{_user.username}}\"}}"}}}]}),
        ),
        ("broken_json", json!({"indices": [{"names": ["docs"], "privileges": ["read"], "query": "{\"term\":"}]})),
        ("stored_template", json!({"indices": [{"names": ["docs"], "privileges": ["read"], "query": {"template": {"id": "nope"}}}]})),
        ("fls_only", json!({"indices": [{"names": ["docs"], "privileges": ["read"], "field_security": {"grant": ["name", "dept"]}}]})),
        ("writer", json!({"indices": [{"names": ["docs"], "privileges": ["read", "write"], "query": {"term": {"dept": "eng"}}}]})),
    ];
    for (name, body) in roles {
        let (st, r) = es.put(&ADMIN, &format!("/_security/role/{name}"), body).await;
        assert_eq!(st, 200, "role {name}: {r}");
    }
    for (user, roles) in [
        ("dave", json!(["eng_nosalary"])),
        ("sam", json!(["sales_only"])),
        ("alice", json!(["mine"])),
        ("bob", json!(["mine"])),
        ("eve", json!(["broken_json"])),
        ("frank", json!(["stored_template"])),
        ("gina", json!(["fls_only"])),
        ("both", json!(["eng_nosalary", "sales_only"])),
        ("wendy", json!(["writer"])),
    ] {
        let (st, r) = es.put(&ADMIN, &format!("/_security/user/{user}"), json!({"password": PW, "roles": roles})).await;
        assert_eq!(st, 200, "user {user}: {r}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn dls_fls_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    setup(&es).await;
    let dave = As::User("dave");

    // The role API no longer gates DLS/FLS behind a license.
    let (st, r) = es.get(&ADMIN, "/_security/role/eng_nosalary").await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["eng_nosalary"]["indices"][0]["field_security"]["except"], json!(["salary", "private", "obj.secret"]));

    // --- superuser sees everything
    let (st, r) = es.post(&ADMIN, "/docs/_search", json!({"query": {"match_all": {}}})).await;
    assert_eq!(st, 200);
    assert_eq!(ids(&r), ["1", "2", "3"]);
    assert_eq!(r["hits"]["hits"][0]["_source"].as_object().unwrap().contains_key("salary"), true);

    // --- search: DLS filters documents, FLS filters `_source`
    let (st, r) = es.post(&dave, "/docs/_search", json!({"query": {"match_all": {}}, "sort": [{"name": "asc"}]})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(ids(&r), ["1", "2"]);
    assert_eq!(r["hits"]["total"]["value"], 2);
    for h in r["hits"]["hits"].as_array().unwrap() {
        let src = h["_source"].as_object().unwrap();
        assert!(src.contains_key("name") && src.contains_key("dept"), "{h}");
        assert!(!src.contains_key("salary") && !src.contains_key("private"), "{h}");
        assert!(src["obj"].get("secret").is_none() && src["obj"].get("public").is_some(), "{h}");
    }
    // user _source includes apply after FLS, and cannot widen it
    let (_, r) = es.post(&dave, "/docs/_search", json!({"_source": ["salary", "name"]})).await;
    assert_eq!(r["hits"]["hits"][0]["_source"], json!({"name": r["hits"]["hits"][0]["_source"]["name"]}));
    // hidden fields behave as unmapped: no matches, no aggregation values, no leaks via ranges
    for q in [json!({"term": {"salary": 100}}), json!({"range": {"salary": {"gte": 0}}}), json!({"match": {"private": "needle"}}), json!({"exists": {"field": "salary"}}), json!({"query_string": {"query": "needle"}})] {
        let (st, r) = es.post(&dave, "/docs/_search", json!({"query": q})).await;
        assert_eq!(st, 200, "{q}: {r}");
        assert_eq!(r["hits"]["total"]["value"], 0, "{q}: {r}");
    }
    let (st, r) = es.post(&dave, "/docs/_search", json!({"size": 0, "aggs": {"s": {"sum": {"field": "salary"}}, "d": {"terms": {"field": "dept"}}, "ph": {"terms": {"field": "private.keyword"}}}})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["aggregations"]["s"]["value"], 0.0, "{r}");
    assert_eq!(r["aggregations"]["d"]["buckets"], json!([{"key": "eng", "doc_count": 2}]));
    assert_eq!(r["aggregations"]["ph"]["buckets"], json!([]));
    // sorting on a hidden field fails like an unmapped field
    let (st, _) = es.post(&dave, "/docs/_search", json!({"sort": ["salary"]})).await;
    assert_eq!(st, 400);
    // fields / docvalue_fields / stored_fields only return granted fields
    let (_, r) = es.post(&dave, "/docs/_search", json!({"_source": false, "fields": ["*"], "docvalue_fields": ["salary", "dept"]})).await;
    for h in r["hits"]["hits"].as_array().unwrap() {
        let f = h["fields"].as_object().unwrap();
        assert!(f.contains_key("dept") && f.contains_key("name"), "{h}");
        assert!(!f.contains_key("salary") && !f.contains_key("private") && !f.contains_key("obj.secret"), "{h}");
    }
    // highlighting cannot reveal hidden fields
    let (st, r) = es.post(&dave, "/docs/_search", json!({"query": {"match": {"note": "haystack"}}, "highlight": {"require_field_match": false, "fields": {"private": {}, "note": {}}}})).await;
    assert_eq!(st, 200, "{r}");
    for h in r["hits"]["hits"].as_array().unwrap() {
        assert!(h["highlight"].get("private").is_none(), "{h}");
    }
    // inner source filtering through top_hits
    let (_, r) = es.post(&dave, "/docs/_search", json!({"size": 0, "aggs": {"t": {"top_hits": {"size": 5}}}})).await;
    for h in r["aggregations"]["t"]["hits"]["hits"].as_array().unwrap() {
        assert!(h["_source"].get("salary").is_none(), "{h}");
    }
    // features that read outside the filtered paths are refused, never served unfiltered
    let (st, _) = es.post(&dave, "/docs/_search", json!({"script_fields": {"s": {"script": "params._source.salary"}}})).await;
    assert_eq!(st, 403);
    let (st, _) = es.post(&dave, "/docs/_search", json!({"query": {"script": {"script": "doc['salary'].value > 0"}}})).await;
    assert_eq!(st, 403);
    let (st, _) = es.post(&dave, "/docs/_search", json!({"suggest": {"s": {"text": "x", "term": {"field": "name"}}}})).await;
    assert_eq!(st, 403);
    let (st, _) = es.get(&dave, "/docs/_termvectors/3?fields=name").await;
    assert_eq!(st, 403);
    let (st, _) = es.get(&dave, "/docs/_explain/3").await;
    assert_eq!(st, 403);
    // ES|QL supports restricted reads; verify the policy rather than the
    // obsolete blanket denial from before the source restriction hook.
    let (st, result) = es.post(&dave, "/_query", json!({"query": "FROM docs | SORT name"})).await;
    assert_eq!(st, 200, "{result}");
    assert_eq!(result["values"].as_array().unwrap().len(), 2, "ES|QL bypassed DLS: {result}");
    let columns = result["columns"].as_array().unwrap();
    assert!(!columns.iter().any(|c| matches!(c["name"].as_str(), Some("salary" | "private" | "obj.secret"))), "ES|QL bypassed FLS: {result}");
    let name = columns.iter().position(|c| c["name"] == "name").unwrap();
    assert_eq!(result["values"].as_array().unwrap().iter().map(|r| r[name].clone()).collect::<Vec<_>>(), vec![json!("A"), json!("B")]);

    // --- count
    let (_, r) = es.post(&dave, "/docs/_count", json!({})).await;
    assert_eq!(r["count"], 2);

    // --- get / _source / mget
    let (st, r) = es.get(&dave, "/docs/_doc/1").await;
    assert_eq!(st, 200);
    assert!(r["_source"].get("salary").is_none() && r["_source"]["name"] == "A");
    let (st, r) = es.get(&dave, "/docs/_doc/3").await;
    assert_eq!(st, 404, "{r}");
    assert_eq!(r["found"], false);
    let (st, _) = es.get(&dave, "/docs/_source/3").await;
    assert_eq!(st, 404);
    let (st, r) = es.get(&dave, "/docs/_source/2").await;
    assert_eq!(st, 200);
    assert!(r.get("salary").is_none() && r["name"] == "B");
    let (_, r) = es.post(&dave, "/docs/_mget", json!({"ids": ["1", "3"]})).await;
    assert_eq!(r["docs"][0]["found"], true);
    assert!(r["docs"][0]["_source"].get("salary").is_none());
    assert_eq!(r["docs"][1]["found"], false);
    let (_, r) = es.get(&dave, "/docs/_doc/1?stored_fields=salary&_source=false").await;
    assert!(r.get("fields").is_none(), "{r}");

    // --- field_caps and mappings hide ungranted fields
    let (st, r) = es.get(&dave, "/docs/_field_caps?fields=*").await;
    assert_eq!(st, 200, "{r}");
    let f = r["fields"].as_object().unwrap();
    assert!(f.contains_key("name") && f.contains_key("obj.public") && f.contains_key("_id"));
    assert!(!f.contains_key("salary") && !f.contains_key("private") && !f.contains_key("obj.secret"), "{r}");
    let (_, r) = es.get(&dave, "/docs/_mapping").await;
    let props = &r["docs"]["mappings"]["properties"];
    assert!(props.get("salary").is_none() && props.get("private").is_none(), "{r}");
    assert!(props["obj"]["properties"].get("secret").is_none() && props["obj"]["properties"].get("public").is_some(), "{r}");
    let (_, r) = es.get(&dave, "/docs").await;
    assert!(r["docs"]["mappings"]["properties"].get("salary").is_none(), "{r}");
    let (_, r) = es.get(&dave, "/docs/_mapping/field/salary").await;
    assert!(r["docs"]["mappings"].as_object().map(|m| m.is_empty()).unwrap_or(true), "{r}");

    // --- msearch and scroll and PIT carry the restrictions
    let (st, r) = es.ndjson(&dave, "/_msearch", &[json!({"index": "docs"}), json!({"query": {"match_all": {}}}), json!({"index": "docs"}), json!({"query": {"term": {"dept": "sales"}}})]).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["responses"][0]["hits"]["total"]["value"], 2);
    assert_eq!(r["responses"][1]["hits"]["total"]["value"], 0);
    let (_, r) = es.post(&dave, "/docs/_search?scroll=1m", json!({"size": 1, "sort": ["name"]})).await;
    let sid = r["_scroll_id"].as_str().unwrap().to_string();
    let mut seen = ids(&r);
    for _ in 0..5 {
        let (_, r) = es.post(&dave, "/_search/scroll", json!({"scroll": "1m", "scroll_id": sid})).await;
        let page = ids(&r);
        if page.is_empty() {
            break;
        }
        for h in r["hits"]["hits"].as_array().unwrap() {
            assert!(h["_source"].get("salary").is_none());
        }
        seen.extend(page);
    }
    seen.sort();
    assert_eq!(seen, ["1", "2"]);
    let (st, r) = es.post(&dave, "/docs/_pit?keep_alive=1m", json!({})).await;
    assert_eq!(st, 200, "{r}");
    let pit = r["id"].as_str().unwrap().to_string();
    let (st, r) = es.post(&dave, "/_search", json!({"pit": {"id": pit}, "query": {"match_all": {}}})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(ids(&r), ["1", "2"]);
    assert!(r["hits"]["hits"][0]["_source"].get("salary").is_none());

    // --- terms lookup and more_like_this cannot read documents DLS hides
    let (_, r) = es.post(&dave, "/docs/_search", json!({"query": {"terms": {"owner": {"index": "docs", "id": "3", "path": "owner"}}}})).await;
    assert_eq!(r["hits"]["total"]["value"], 0, "{r}");
    let (_, r) = es.post(&dave, "/docs/_search", json!({"query": {"terms": {"owner": {"index": "docs", "id": "1", "path": "owner"}}}})).await;
    assert_eq!(ids(&r), ["1"]);
    let (_, r) = es.post(&dave, "/docs/_search", json!({"query": {"terms": {"owner": {"index": "docs", "id": "1", "path": "salary"}}}})).await;
    assert_eq!(r["hits"]["total"]["value"], 0, "{r}");

    // --- writes that would read and rewrite hidden data are refused
    let wendy = As::User("wendy");
    let (st, r) = es.post(&wendy, "/docs/_update/1", json!({"doc": {"name": "Z"}})).await;
    assert_eq!(st, 400, "{r}");
    assert!(r["error"]["reason"].as_str().unwrap().contains("field or document level security"), "{r}");
    let (st, r) = es.ndjson(&wendy, "/_bulk?refresh=true", &[json!({"update": {"_index": "docs", "_id": "1"}}), json!({"doc": {"name": "Z"}}), json!({"index": {"_index": "docs", "_id": "9"}}), json!({"name": "N", "dept": "eng"})]).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["items"][0]["update"]["status"], 400, "{r}");
    assert_eq!(r["items"][1]["index"]["status"], 201, "{r}");
    let (st, _) = es.post(&wendy, "/docs/_update_by_query", json!({"script": "ctx._source.name = 'x'"})).await;
    assert_eq!(st, 403);
    let (st, _) = es.post(&wendy, "/_reindex", json!({"source": {"index": "docs"}, "dest": {"index": "copy"}})).await;
    assert_eq!(st, 403);
    let (_, r) = es.get(&ADMIN, "/docs/_doc/1").await;
    assert_eq!(r["_source"]["name"], "A", "refused update must not have been applied");
    let (st, _) = es.put(&wendy, "/docs/_alias/leak", json!({})).await;
    assert_eq!(st, 403);

    // --- multiple roles union their documents; a plain subject is unaffected
    let both = As::User("both");
    let (_, r) = es.post(&both, "/docs/_search", json!({})).await;
    assert_eq!(ids(&r), ["1", "2", "3", "9"]);
    assert!(r["hits"]["hits"].as_array().unwrap().iter().any(|h| h["_source"].get("salary").is_some()), "the sales_only role grants every field of its documents");
    let (_, r) = es.post(&As::User("sam"), "/docs/_search", json!({})).await;
    assert_eq!(ids(&r), ["3"]);

    // --- template DLS substitutes the user
    for (u, want) in [("alice", "1"), ("bob", "2")] {
        let (st, r) = es.post(&As::User(u), "/docs/_search", json!({})).await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(ids(&r), [want]);
    }

    // --- FLS only
    let (_, r) = es.post(&As::User("gina"), "/docs/_search", json!({})).await;
    assert_eq!(ids(&r), ["1", "2", "3", "9"]);
    assert_eq!(r["hits"]["hits"][0]["_source"].as_object().unwrap().len(), 2);

    // --- default-deny: a filter that cannot be computed denies, never returns unfiltered data
    for u in ["eve", "frank"] {
        let (st, r) = es.post(&As::User(u), "/docs/_search", json!({})).await;
        assert_eq!(st, 403, "{u}: {r}");
        assert!(r.get("hits").is_none());
        let (st, _) = es.get(&As::User(u), "/docs/_doc/1").await;
        assert_eq!(st, 403, "{u}");
        let (st, _) = es.post(&As::User(u), "/docs/_count", json!({})).await;
        assert_eq!(st, 403, "{u}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn api_key_intersects_with_owner() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    setup(&es).await;
    let dave = As::User("dave");
    // dave (eng docs, no salary/private/obj.secret) creates a key narrowing fields further.
    let (st, r) = es
        .post(&dave, "/_security/api_key", json!({"name": "k", "role_descriptors": {"r": {"indices": [{"names": ["docs"], "privileges": ["read"], "field_security": {"grant": ["name", "salary", "owner"]}}]}}}))
        .await;
    assert_eq!(st, 200, "{r}");
    let key = As::Key(r["encoded"].as_str().unwrap().to_string());
    let (st, r) = es.post(&key, "/docs/_search", json!({})).await;
    assert_eq!(st, 200, "{r}");
    // the owner's document filter still applies
    assert_eq!(ids(&r), ["1", "2"]);
    for h in r["hits"]["hits"].as_array().unwrap() {
        let keys: Vec<&String> = h["_source"].as_object().unwrap().keys().collect();
        assert_eq!(keys.len(), 2, "name + owner only (salary is denied by the owner role): {h}");
        assert!(h["_source"].get("name").is_some() && h["_source"].get("owner").is_some());
    }
    let (st, _) = es.get(&key, "/docs/_doc/3").await;
    assert_eq!(st, 404);

    // A key with its own DLS on top of an unrestricted owner.
    let (st, r) = es
        .post(&ADMIN, "/_security/api_key", json!({"name": "k2", "role_descriptors": {"r": {"indices": [{"names": ["docs"], "privileges": ["read"], "query": {"term": {"dept": "sales"}}, "field_security": {"grant": ["name", "dept"]}}]}}}))
        .await;
    assert_eq!(st, 200, "{r}");
    let key2 = As::Key(r["encoded"].as_str().unwrap().to_string());
    let (_, r) = es.post(&key2, "/docs/_search", json!({})).await;
    assert_eq!(ids(&r), ["3"]);
    assert_eq!(r["hits"]["hits"][0]["_source"], json!({"name": "C", "dept": "sales"}));
}

#[tokio::test]
async fn application_privilege_expansion_preserves_union_scope_and_updates() {
    let dir = tempfile::tempdir().unwrap();
    let (es, _) = start(dir.path()).await;
    let admin = As::User("admin");
    let definitions = json!({"test-app": {
        "both": {"actions": ["data:read/*", "data:write/*"]},
        "read": {"actions": ["data:read/*"]},
        "write": {"actions": ["data:write/*"]}
    }});
    assert_eq!(es.post(&admin, "/_security/privilege", definitions).await.0, 200);
    let role = json!({"applications": [
        {"application": "test-app", "resources": ["space:one"], "privileges": ["read"]},
        {"application": "test-app", "resources": ["space:one"], "privileges": ["write", "undeclared"]},
        {"application": "other-app", "resources": ["space:two"], "privileges": ["*"]}
    ]});
    assert_eq!(es.put(&admin, "/_security/role/app-check", role).await.0, 200);
    assert_eq!(es.put(&admin, "/_security/user/app-user", json!({"password": PW, "roles": ["app-check"]})).await.0, 200);
    let body = json!({"application": [{"application": "test-app", "resources": ["space:one", "space:two"], "privileges": ["both", "undeclared", "absent"]}]});
    let user = As::User("app-user");
    let (status, result) = es.post(&user, "/_security/user/_has_privileges", body.clone()).await;
    assert_eq!(status, 200, "{result}");
    assert_eq!(result["application"]["test-app"]["space:one"], json!({"both": true, "undeclared": true, "absent": false}));
    assert_eq!(result["application"]["test-app"]["space:two"], json!({"both": false, "undeclared": false, "absent": false}));
    // A later request must see changed definitions, including a formerly unknown name.
    assert_eq!(es.post(&admin, "/_security/privilege", json!({"test-app": {
        "both": {"actions": ["data:admin/*"]},
        "undeclared": {"actions": ["data:other/*"]}
    }})).await.0, 200);
    let (status, result) = es.post(&user, "/_security/user/_has_privileges", body).await;
    assert_eq!(status, 200, "{result}");
    assert_eq!(result["application"]["test-app"]["space:one"]["both"], false);
    assert_eq!(result["application"]["test-app"]["space:one"]["undeclared"], true);
}
