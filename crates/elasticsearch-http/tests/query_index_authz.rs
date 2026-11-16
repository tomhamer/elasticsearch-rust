//! Index authorization of the query languages (ES|QL, SQL, EQL) and of the
//! other reads whose indices are not named in the URL path, over a
//! security-enabled node: a user or API key with `read` on `public` only
//! must never read `private`, whatever expression, alias, data stream,
//! join, cursor or stored result the request goes through.

use base64::Engine;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::security::{hash::Hasher, SecConfig};
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;

const PW: &str = "query-authz-test-password";

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
    (Es { base: format!("http://{addr}"), client: Client::builder().timeout(std::time::Duration::from_secs(30)).build().unwrap(), _task: task }, node)
}

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
    async fn esql(&self, w: &As, q: &str) -> (u16, Value) {
        self.post(w, "/_query", json!({"query": q})).await
    }
    async fn sql(&self, w: &As, q: &str) -> (u16, Value) {
        self.post(w, "/_sql?format=json", json!({"query": q})).await
    }
}

const ADMIN: As = As::User("admin");
/// `read` on `public` and `lookup-public` only.
const READER: As = As::User("reader");
/// `read` on `public`, `view_index_metadata` (not `read`) on `private`.
const META: As = As::User("meta");
/// `monitor_ml` only, no index privileges.
const ML: As = As::User("mlmon");
/// `read` on `public` plus the `monitor_enrich` cluster privilege.
const ENRICHER: As = As::User("enricher");

fn mapping() -> Value {
    json!({"properties": {
        "@timestamp": {"type": "date"},
        "event": {"properties": {"category": {"type": "keyword"}}},
        "k": {"type": "keyword"},
        "v": {"type": "long"},
        "secret": {"type": "keyword"},
        "msg": {"type": "text", "term_vector": "with_positions_offsets"}
    }})
}

async fn setup(es: &Es) -> As {
    for index in ["public", "private", "private-2"] {
        let (st, r) = es.put(&ADMIN, &format!("/{index}"), json!({"settings": {"number_of_shards": 1}, "mappings": mapping()})).await;
        assert_eq!(st, 200, "{index}: {r}");
    }
    for (index, field) in [("lookup-public", "label"), ("lookup-private", "secret")] {
        let (st, r) = es.put(&ADMIN, &format!("/{index}"), json!({"settings": {"index.mode": "lookup", "number_of_shards": 1}, "mappings": {"properties": {"k": {"type": "keyword"}, field: {"type": "keyword"}}}})).await;
        assert_eq!(st, 200, "{index}: {r}");
    }
    let (st, r) = es.put(&ADMIN, "/_index_template/logs-priv", json!({"index_patterns": ["logs-priv*"], "data_stream": {}, "template": {"mappings": mapping()}})).await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = es.call(&ADMIN, Method::PUT, "/_data_stream/logs-priv", None).await;
    assert_eq!(st, 200, "{r}");
    let ts = "2026-11-01T00:00:00Z";
    let doc = |v: i64, secret: &str| json!({"@timestamp": ts, "event": {"category": "process"}, "k": "a", "v": v, "secret": secret, "msg": format!("{secret} words")});
    let lines = vec![
        json!({"index": {"_index": "public", "_id": "1"}}), doc(1, "public-data"),
        json!({"index": {"_index": "public", "_id": "2"}}), doc(1, "public-data"),
        json!({"index": {"_index": "private", "_id": "1"}}), doc(2, "top-secret"),
        json!({"index": {"_index": "private", "_id": "2"}}), doc(2, "top-secret"),
        json!({"index": {"_index": "private-2", "_id": "1"}}), doc(3, "top-secret-2"),
        json!({"index": {"_index": "lookup-public", "_id": "1"}}), json!({"k": "a", "label": "visible"}),
        json!({"index": {"_index": "lookup-private", "_id": "1"}}), json!({"k": "a", "secret": "joined-secret"}),
        json!({"create": {"_index": "logs-priv"}}), doc(4, "stream-secret"),
    ];
    let (st, r) = es.ndjson(&ADMIN, "/_bulk?refresh=true", &lines).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["errors"], false, "{r}");
    let (st, r) = es.post(&ADMIN, "/_aliases", json!({"actions": [{"add": {"index": "private", "alias": "priv-alias"}}]})).await;
    assert_eq!(st, 200, "{r}");

    let roles = [
        ("pub_read", json!({"indices": [{"names": ["public", "lookup-public"], "privileges": ["read"]}]})),
        ("pub_meta", json!({"indices": [{"names": ["public"], "privileges": ["read"]}, {"names": ["private"], "privileges": ["view_index_metadata"]}]})),
        ("ml_monitor", json!({"cluster": ["monitor_ml"]})),
        ("pub_enrich", json!({"cluster": ["monitor_enrich"], "indices": [{"names": ["public"], "privileges": ["read"]}]})),
        ("pub_write", json!({"indices": [{"names": ["public", "copy"], "privileges": ["read", "write", "create_index"]}]})),
        ("esql_monitor", json!({"cluster": ["monitor_esql"], "indices": [{"names": ["public"], "privileges": ["read"]}]})),
        ("alias_read", json!({"indices": [{"names": ["priv-alias"], "privileges": ["read"]}]})),
        ("ds_data", json!({"indices": [{"names": ["logs-priv"], "privileges": ["read"]}]})),
        ("ds_failures", json!({"indices": [{"names": ["logs-priv"], "privileges": ["read_failure_store"]}]})),
    ];
    for (name, body) in roles {
        let (st, r) = es.put(&ADMIN, &format!("/_security/role/{name}"), body).await;
        assert_eq!(st, 200, "role {name}: {r}");
    }
    for (user, role) in [("reader", "pub_read"), ("meta", "pub_meta"), ("mlmon", "ml_monitor"), ("enricher", "pub_enrich"), ("writer", "pub_write"), ("esqlmon", "esql_monitor"), ("aliasreader", "alias_read"), ("dsdata", "ds_data"), ("dsfailures", "ds_failures")] {
        let (st, r) = es.put(&ADMIN, &format!("/_security/user/{user}"), json!({"password": PW, "roles": [role]})).await;
        assert_eq!(st, 200, "user {user}: {r}");
    }
    // An API key of the superuser, limited to `read` on `public`.
    let (st, r) = es
        .post(&ADMIN, "/_security/api_key", json!({"name": "pub", "role_descriptors": {"r": {"indices": [{"names": ["public", "lookup-public"], "privileges": ["read"]}]}}}))
        .await;
    assert_eq!(st, 200, "{r}");
    As::Key(r["encoded"].as_str().unwrap().to_string())
}

fn text(v: &Value) -> String {
    v.to_string()
}

/// No response may carry any private value.
fn assert_no_leak(what: &str, r: &Value) {
    let s = text(r);
    for secret in ["top-secret", "stream-secret", "joined-secret", "\"v\":2", "\"v\":3"] {
        assert!(!s.contains(secret), "{what} leaked {secret}: {s}");
    }
}

fn assert_unknown_index(what: &str, st: u16, r: &Value, name: &str) {
    assert_eq!(st, 400, "{what}: {r}");
    assert_eq!(r["error"]["type"], "verification_exception", "{what}: {r}");
    assert!(r["error"]["reason"].as_str().unwrap_or("").contains(&format!("Unknown index [{name}]")), "{what}: {r}");
    assert_no_leak(what, r);
}

fn assert_security_exception(what: &str, st: u16, r: &Value) {
    assert_eq!(st, 403, "{what}: {r}");
    assert_eq!(r["error"]["type"], "security_exception", "{what}: {r}");
    assert_no_leak(what, r);
}

/// The `_index` values of an ES|QL result whose first column is `_index`.
fn esql_indices(r: &Value) -> Vec<String> {
    let mut out: Vec<String> = r["values"].as_array().unwrap().iter().map(|row| row[0].as_str().unwrap().to_string()).collect();
    out.sort();
    out.dedup();
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn esql_reads_only_authorized_indices() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let key = setup(&es).await;

    // The superuser reads everything.
    let (st, r) = es.esql(&ADMIN, "FROM private | STATS n = COUNT(*)").await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["values"], json!([[2]]));
    let (st, r) = es.esql(&ADMIN, "FROM * METADATA _index | KEEP _index").await;
    assert_eq!(st, 200, "{r}");
    assert!(esql_indices(&r).contains(&"private".to_string()), "{r}");

    for who in [READER, key.clone()] {
        // Authorized access works.
        let (st, r) = es.esql(&who, "FROM public | STATS n = COUNT(*)").await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["values"], json!([[2]]));
        let (st, r) = es.esql(&who, "FROM public | LOOKUP JOIN lookup-public ON k | KEEP label | LIMIT 1").await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["values"], json!([["visible"]]));

        // Explicit unauthorized names (indices, aliases, data streams,
        // selectors, backing indices) resolve to nothing, as ES's lenient
        // field-caps resolution leaves them.
        for (q, name) in [
            ("FROM private | LIMIT 10", "private"),
            ("FROM priv-alias | LIMIT 10", "priv-alias"),
            ("FROM logs-priv | LIMIT 10", "logs-priv"),
            ("FROM priv* | LIMIT 10", "priv*"),
            ("FROM logs-* | LIMIT 10", "logs-*"),
            ("FROM .ds-logs-priv* | LIMIT 10", ".ds-logs-priv*"),
        ] {
            let (st, r) = es.esql(&who, q).await;
            assert_unknown_index(q, st, &r, name);
        }
        // Component selectors name their index.
        // (A selector contains `:` but is never a remote cluster part: it is
        // authorized here, alone or in a list.)
        for q in ["FROM private::data | LIMIT 10", "FROM logs-priv::failures | LIMIT 10", "FROM logs-priv::data | LIMIT 10", "FROM public,private::data | LIMIT 10", "FROM public::data,logs-priv::failures | LIMIT 10"] {
            let (st, r) = es.esql(&who, q).await;
            assert_eq!(st, 400, "{q}: {r}");
            assert_no_leak(q, &r);
        }
        // An unauthorized explicit name fails the whole list (ES does not
        // silently drop it), naming only the unauthorized names.
        for (q, name) in [("FROM public,private | LIMIT 10", "private"), ("FROM p*,priv-alias | LIMIT 10", "priv-alias"), ("FROM public,private,priv-alias | LIMIT 10", "private,priv-alias")] {
            let (st, r) = es.esql(&who, q).await;
            assert_unknown_index(q, st, &r, name);
        }
        // Wildcards expand to the authorized indices only.
        for q in ["FROM * METADATA _index | KEEP _index", "FROM p* METADATA _index | KEEP _index", "FROM *,-public METADATA _index | KEEP _index"] {
            let (st, r) = es.esql(&who, q).await;
            if q.contains("-public") {
                assert_eq!(st, 200, "{q}: {r}");
                assert_eq!(esql_indices(&r), ["lookup-public"], "{q}: {r}");
            } else {
                assert_eq!(st, 200, "{q}: {r}");
                for i in esql_indices(&r) {
                    assert!(i == "public" || i == "lookup-public", "{q} read {i}: {r}");
                }
            }
            assert_no_leak(q, &r);
        }
        let (st, r) = es.esql(&who, "FROM * | STATS m = MAX(v)").await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["values"], json!([[1]]), "{r}");
        // A lookup join on an unauthorized index.
        let (st, r) = es.esql(&who, "FROM public | LOOKUP JOIN lookup-private ON k | KEEP secret").await;
        assert_unknown_index("lookup join", st, &r, "lookup-private");
        // Subqueries and the async endpoint resolve the same way.
        let (st, r) = es.post(&who, "/_query/async", json!({"query": "FROM private | LIMIT 10", "wait_for_completion_timeout": "30s"})).await;
        assert_unknown_index("async", st, &r, "private");
        let (st, r) = es.post(&who, "/_query/async", json!({"query": "FROM public | STATS n = COUNT(*)", "wait_for_completion_timeout": "30s"})).await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["values"], json!([[2]]));
    }

    // A name the caller may describe (view_index_metadata) but not read is
    // unknown too, and wildcards skip it (ES `EsqlSecurityIT.testInsufficientPrivilege`).
    let (st, r) = es.esql(&META, "FROM private | STATS n = COUNT(*)").await;
    assert_unknown_index("metadata only", st, &r, "private");
    let (st, r) = es.esql(&META, "FROM p* METADATA _index | KEEP _index").await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(esql_indices(&r), ["public"], "{r}");

    // ENRICH reads the policy's enrich index (built from `private` by an
    // enrich admin) through ES's enrich lookup, which needs `monitor_enrich`.
    let (st, r) = es.put(&ADMIN, "/_enrich/policy/priv-policy", json!({"match": {"indices": "private", "match_field": "k", "enrich_fields": ["secret"]}})).await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = es.call(&ADMIN, Method::PUT, "/_enrich/policy/priv-policy/_execute", None).await;
    assert_eq!(st, 200, "{r}");
    let enrich = "FROM public | ENRICH priv-policy ON k WITH secret | KEEP secret | LIMIT 1";
    for who in [READER, key.clone()] {
        let (st, r) = es.esql(&who, enrich).await;
        assert_security_exception("enrich", st, &r);
        assert!(r["error"]["reason"].as_str().unwrap().contains("doesn't have sufficient privileges to perform enrich lookup: privilege [monitor_enrich] is missing"), "{r}");
    }
    let (st, r) = es.esql(&ENRICHER, enrich).await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = es.esql(&ENRICHER, "FROM private | LIMIT 1").await;
    assert_unknown_index("enricher", st, &r, "private");

    // A user without any index privilege cannot run a query at all, async included.
    for path in ["/_query", "/_query/async"] {
        let (st, r) = es.post(&ML, path, json!({"query": "FROM public | LIMIT 1"})).await;
        assert_security_exception(path, st, &r);
    }

    // Stored async results belong to their owner.
    let (st, r) = es.post(&ADMIN, "/_query/async", json!({"query": "FROM private | KEEP secret | LIMIT 1", "keep_on_completion": true, "wait_for_completion_timeout": "30s"})).await;
    assert_eq!(st, 200, "{r}");
    let id = r["id"].as_str().unwrap().to_string();
    for who in [READER, key.clone()] {
        let (st, r) = es.get(&who, &format!("/_query/async/{id}")).await;
        assert_eq!(st, 404, "{r}");
        assert_no_leak("async get", &r);
        let (st, r) = es.call(&who, Method::POST, &format!("/_query/async/{id}/stop"), None).await;
        assert_eq!(st, 404, "{r}");
        let (st, r) = es.call(&who, Method::DELETE, &format!("/_query/async/{id}"), None).await;
        assert_eq!(st, 404, "{r}");
    }
    // Without any index privilege the stored-result endpoints are refused.
    for (m, path) in [(Method::GET, format!("/_query/async/{id}")), (Method::POST, format!("/_query/async/{id}/stop")), (Method::DELETE, format!("/_query/async/{id}"))] {
        let (st, r) = es.call(&ML, m, &path, None).await;
        assert_security_exception(&path, st, &r);
    }
    let (st, r) = es.get(&ADMIN, &format!("/_query/async/{id}")).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["values"], json!([["top-secret"]]));
    assert!(r.get("__owner").is_none(), "{r}");
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_reads_only_authorized_indices() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let key = setup(&es).await;

    let (st, r) = es.sql(&ADMIN, "SELECT COUNT(*) AS n FROM private").await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["rows"], json!([[2]]));

    for who in [READER, key.clone()] {
        let (st, r) = es.sql(&who, "SELECT COUNT(*) AS n FROM public").await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["rows"], json!([[2]]));
        for (q, name) in [
            ("SELECT * FROM private", "private"),
            ("SELECT secret FROM private WHERE v > 0", "private"),
            ("SELECT * FROM \"priv-alias\"", "priv-alias"),
            ("SELECT * FROM \"logs-priv\"", "logs-priv"),
            ("SELECT * FROM \"priv*\"", "priv*"),
        ] {
            let (st, r) = es.sql(&who, q).await;
            assert_unknown_index(q, st, &r, name);
        }
        let (st, r) = es.sql(&who, "SELECT MAX(v) AS m FROM \"*\"").await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["rows"], json!([[1]]), "{r}");
        let (st, r) = es.sql(&who, "SELECT v FROM \"public,private\"").await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["rows"], json!([[1], [1]]), "{r}");
        // Metadata commands list and describe only what the caller may see.
        let (st, r) = es.sql(&who, "SHOW TABLES").await;
        assert_eq!(st, 200, "{r}");
        let names: Vec<&str> = r["rows"].as_array().unwrap().iter().map(|row| row[1].as_str().unwrap()).collect();
        assert!(names.contains(&"public"), "{r}");
        for hidden in ["private", "private-2", "priv-alias", "lookup-private"] {
            assert!(!names.contains(&hidden), "SHOW TABLES listed {hidden}: {r}");
        }
        // (ES describes an index the caller may not see as no columns)
        let (st, r) = es.sql(&who, "SHOW COLUMNS FROM private").await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["rows"], json!([]), "{r}");
        let (st, r) = es.post(&who, "/_sql/translate", json!({"query": "SELECT secret FROM private WHERE v = 2"})).await;
        assert_eq!(st, 400, "{r}");
        let (st, r) = es.post(&who, "/_sql/translate", json!({"query": "SELECT v FROM public WHERE v = 1"})).await;
        assert_eq!(st, 200, "{r}");
        // Async SQL resolves the same way.
        let (st, r) = es.post(&who, "/_sql?format=json", json!({"query": "SELECT * FROM private", "wait_for_completion_timeout": "30s"})).await;
        assert_unknown_index("async sql", st, &r, "private");
    }

    let (st, r) = es.sql(&META, "SELECT COUNT(*) FROM private").await;
    assert_security_exception("metadata only", st, &r);
    assert!(r["error"]["reason"].as_str().unwrap().contains("on indices [private]"), "{r}");
    // `view_index_metadata` is enough to describe the index.
    let (st, r) = es.sql(&META, "SHOW COLUMNS FROM private").await;
    assert_eq!(st, 200, "{r}");

    // Cursors and stored async results belong to their owner.
    let (st, first) = es.post(&ADMIN, "/_sql?format=json", json!({"query": "SELECT secret FROM private", "fetch_size": 1})).await;
    assert_eq!(st, 200, "{first}");
    let cursor = first["cursor"].as_str().unwrap().to_string();
    for who in [READER, key.clone()] {
        let (st, r) = es.post(&who, "/_sql?format=json", json!({"cursor": cursor})).await;
        assert_eq!(st, 404, "{r}");
        assert_no_leak("cursor", &r);
        let (st, r) = es.post(&who, "/_sql/close", json!({"cursor": cursor})).await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["succeeded"], false, "{r}");
    }
    let (st, r) = es.post(&ADMIN, "/_sql?format=json", json!({"cursor": cursor})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["rows"], json!([["top-secret"]]));
    let (st, r) = es.post(&ADMIN, "/_sql?format=json", json!({"query": "SELECT secret FROM private LIMIT 1", "wait_for_completion_timeout": "30s", "keep_on_completion": true})).await;
    assert_eq!(st, 200, "{r}");
    let id = r["id"].as_str().unwrap().to_string();
    for who in [READER, key.clone()] {
        for path in [format!("/_sql/async/{id}?format=json"), format!("/_sql/async/status/{id}")] {
            let (st, r) = es.get(&who, &path).await;
            assert_eq!(st, 404, "{path}: {r}");
            assert_no_leak(&path, &r);
        }
        let (st, r) = es.call(&who, Method::DELETE, &format!("/_sql/async/delete/{id}"), None).await;
        assert_eq!(st, 404, "{r}");
    }
    let (st, r) = es.get(&ADMIN, &format!("/_sql/async/{id}?format=json")).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["rows"], json!([["top-secret"]]));
}

#[tokio::test(flavor = "multi_thread")]
async fn eql_reads_only_authorized_indices() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let key = setup(&es).await;
    let q = json!({"query": "process where true"});
    for who in [READER, key.clone()] {
        let (st, r) = es.post(&who, "/public/_eql/search", q.clone()).await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["hits"]["total"]["value"], 2, "{r}");
        for path in ["/private/_eql/search", "/priv-alias/_eql/search", "/logs-priv/_eql/search", "/public,private/_eql/search", "/private::data/_eql/search"] {
            let (st, r) = es.post(&who, path, q.clone()).await;
            assert_security_exception(path, st, &r);
        }
        // Wildcards expand to authorized names only.
        for path in ["/*/_eql/search", "/p*/_eql/search", "/priv*/_eql/search"] {
            let (st, r) = es.post(&who, path, q.clone()).await;
            assert_no_leak(path, &r);
            if st == 200 {
                for e in r["hits"]["events"].as_array().unwrap() {
                    assert_eq!(e["_index"], "public", "{path}: {r}");
                }
            }
        }
    }
    // Stored async results belong to their owner.
    let (st, r) = es.post(&ADMIN, "/private/_eql/search", json!({"query": "process where true", "wait_for_completion_timeout": "30s", "keep_on_completion": true})).await;
    assert_eq!(st, 200, "{r}");
    let id = r["id"].as_str().unwrap().to_string();
    for who in [READER, key.clone()] {
        for path in [format!("/_eql/search/{id}"), format!("/_eql/search/status/{id}")] {
            let (st, r) = es.get(&who, &path).await;
            assert_eq!(st, 404, "{path}: {r}");
            assert_no_leak(&path, &r);
        }
        let (st, r) = es.call(&who, Method::DELETE, &format!("/_eql/search/{id}"), None).await;
        assert_eq!(st, 404, "{r}");
    }
    let (st, r) = es.get(&ADMIN, &format!("/_eql/search/{id}")).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["hits"]["total"]["value"], 2, "{r}");
    assert!(r.get("__owner").is_none(), "{r}");
}

#[tokio::test(flavor = "multi_thread")]
async fn other_body_named_reads_are_authorized() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let key = setup(&es).await;

    // A point in time and a scroll opened by the superuser over `private`.
    let (st, pit) = es.call(&ADMIN, Method::POST, "/private/_pit?keep_alive=1m", None).await;
    assert_eq!(st, 200, "{pit}");
    let (st, scroll) = es.post(&ADMIN, "/private/_search?scroll=1m", json!({"size": 1})).await;
    assert_eq!(st, 200, "{scroll}");
    let scroll_id = scroll["_scroll_id"].as_str().unwrap().to_string();

    for who in [READER, key.clone()] {
        // _mtermvectors items name their own index.
        let (st, r) = es.post(&who, "/public/_mtermvectors", json!({"docs": [{"_index": "private", "_id": "1", "fields": ["msg"]}, {"_id": "1", "fields": ["msg"]}]})).await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["docs"][0]["error"]["type"], "security_exception", "{r}");
        assert_eq!(r["docs"][1]["found"], true, "{r}");
        assert_no_leak("mtermvectors", &r);
        // _mget items naming another index.
        let (st, r) = es.post(&who, "/public/_mget", json!({"docs": [{"_index": "private", "_id": "1"}, {"_id": "1"}]})).await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["docs"][0]["error"]["type"], "security_exception", "{r}");
        assert_no_leak("mget", &r);
        // Another subject's point in time and scroll.
        let (st, r) = es.post(&who, "/_search", json!({"pit": {"id": pit["id"]}})).await;
        assert_security_exception("pit", st, &r);
        // (a scroll is its owner's: another subject does not find it)
        let (st, r) = es.post(&who, "/_search/scroll", json!({"scroll_id": scroll_id, "scroll": "1m"})).await;
        assert_eq!(st, 404, "{r}");
        assert_no_leak("scroll", &r);
        // A search selector on an unauthorized index.
        let (st, r) = es.post(&who, "/private::data/_search", json!({})).await;
        assert_security_exception("selector search", st, &r);
        let (st, r) = es.post(&who, "/public::data/_search", json!({})).await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(r["hits"]["total"]["value"], 2, "{r}");
    }
    // The owner keeps using them.
    let (st, r) = es.post(&ADMIN, "/_search", json!({"pit": {"id": pit["id"]}})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["hits"]["total"]["value"], 2, "{r}");
    let (st, r) = es.post(&ADMIN, "/_search/scroll", json!({"scroll_id": scroll_id, "scroll": "1m"})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["hits"]["hits"].as_array().unwrap().len(), 1, "{r}");

    // A datafeed preview searches its indices: not a monitor_ml action.
    let (st, r) = es.post(&ML, "/_ml/datafeeds/_preview", json!({"datafeed_config": {"indices": ["private"], "query": {"match_all": {}}}, "job_config": {"analysis_config": {"bucket_span": "1h", "detectors": [{"function": "count"}]}, "data_description": {"time_field": "@timestamp"}}})).await;
    assert_security_exception("datafeed preview", st, &r);
    assert!(r["error"]["reason"].as_str().unwrap().contains("cluster:admin/xpack/ml/datafeeds/preview"), "{r}");
}

#[tokio::test(flavor = "multi_thread")]
async fn lookups_scrolls_running_queries_aliases_and_failure_store() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    setup(&es).await;
    let writer = As::User("writer");

    // Terms lookup, more_like_this and percolate-style document lookups read
    // the referenced index as the caller: search, by-query and reindex.
    let lookup_private = json!({"terms": {"secret": {"index": "private", "id": "1", "path": "secret"}}});
    let lookup_public = json!({"terms": {"secret": {"index": "public", "id": "1", "path": "secret"}}});
    let mlt_private = json!({"more_like_this": {"fields": ["msg"], "like": [{"_index": "private", "_id": "1"}], "min_term_freq": 1, "min_doc_freq": 1}});
    for q in [&lookup_private, &mlt_private] {
        let (st, r) = es.post(&writer, "/public/_search", json!({"query": q})).await;
        assert_security_exception("search lookup", st, &r);
    }
    let (st, r) = es.post(&writer, "/public/_search", json!({"query": lookup_public})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["hits"]["total"]["value"], 2, "{r}");
    for path in ["/public/_update_by_query?refresh=true", "/public/_delete_by_query?refresh=true"] {
        let (st, r) = es.post(&writer, path, json!({"query": lookup_private})).await;
        assert_ne!(st, 200, "{path}: {r}");
        assert!(r.to_string().contains("security_exception"), "{path}: {r}");
        assert_no_leak(path, &r);
    }
    let (st, r) = es.post(&writer, "/_reindex?refresh=true", json!({"source": {"index": "public", "query": lookup_private}, "dest": {"index": "copy"}})).await;
    assert_ne!(st, 200, "{r}");
    assert!(r.to_string().contains("security_exception"), "{r}");
    let (st, r) = es.post(&ADMIN, "/public/_count", json!({})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["count"], 2, "nothing was deleted: {r}");
    let (st, r) = es.post(&writer, "/public/_update_by_query?refresh=true", json!({"query": lookup_public})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["updated"], 2, "{r}");

    // Scrolls: a named scroll is cleared only by its owner; clearing all
    // needs the cluster privilege.
    let (st, scroll) = es.post(&ADMIN, "/private/_search?scroll=1m", json!({"size": 1})).await;
    assert_eq!(st, 200, "{scroll}");
    let id = scroll["_scroll_id"].as_str().unwrap().to_string();
    let (st, r) = es.call(&READER, Method::DELETE, "/_search/scroll", Some(json!({"scroll_id": id}))).await;
    assert_eq!(st, 404, "{r}");
    assert_eq!(r["num_freed"], 0, "{r}");
    let (st, r) = es.call(&READER, Method::DELETE, "/_search/scroll/_all", None).await;
    assert_security_exception("clear all", st, &r);
    assert!(r["error"]["reason"].as_str().unwrap().contains("cluster:admin/indices/scroll/clear_all"), "{r}");
    let (st, r) = es.post(&ADMIN, "/_search/scroll", json!({"scroll_id": id, "scroll": "1m"})).await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = es.call(&ADMIN, Method::DELETE, "/_search/scroll/_all", None).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["num_freed"], 1, "{r}");

    // Running ES|QL queries: monitor_esql (or monitor, manage, all).
    for path in ["/_query/queries", "/_query/queries/abc"] {
        let (st, r) = es.get(&READER, path).await;
        assert_security_exception(path, st, &r);
        assert!(r["error"]["reason"].as_str().unwrap().contains("this action is granted by the cluster privileges [monitor_esql,monitor,manage,all]"), "{r}");
    }
    let (st, r) = es.get(&As::User("esqlmon"), "/_query/queries").await;
    assert_eq!(st, 200, "{r}");

    // An alias the caller may read plus its own index the caller may not:
    // ES resolves the index through the alias and refuses its read (403).
    let alias = As::User("aliasreader");
    let (st, r) = es.esql(&alias, "FROM priv-alias | STATS n = COUNT(*)").await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["values"], json!([[2]]));
    let (st, r) = es.esql(&alias, "FROM priv-alias,private | STATS n = COUNT(*)").await;
    assert_security_exception("alias + index", st, &r);
    assert!(r["error"]["reason"].as_str().unwrap().contains("is unauthorized for user [aliasreader] with effective roles [alias_read] on indices [private]"), "{r}");
    let (st, r) = es.esql(&alias, "FROM priv-alias,private-2 | STATS n = COUNT(*)").await;
    assert_unknown_index("alias + other index", st, &r, "private-2");

    // The failure store: `read` reads a data stream's data only,
    // `read_failure_store` its failures only.
    let data = As::User("dsdata");
    let failures = As::User("dsfailures");
    let (st, r) = es.post(&data, "/logs-priv/_search", json!({})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["hits"]["total"]["value"], 1, "{r}");
    let (st, r) = es.post(&data, "/logs-priv::failures/_search", json!({})).await;
    assert_security_exception("data reader on failures", st, &r);
    let (st, r) = es.post(&failures, "/logs-priv/_search", json!({})).await;
    assert_security_exception("failures reader on data", st, &r);
    let (st, r) = es.esql(&failures, "FROM logs-priv | LIMIT 1").await;
    assert_unknown_index("failures reader esql", st, &r, "logs-priv");
    let (st, r) = es.post(&failures, "/logs-priv::failures/_search", json!({})).await;
    assert_ne!(st, 403, "{r}");
    assert_no_leak("failures reader", &r);
}
