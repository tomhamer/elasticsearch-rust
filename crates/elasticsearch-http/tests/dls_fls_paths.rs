//! Document- and field-level security on the less common read paths, and
//! write authorization of ingest reroutes:
//! - FLS on encoded `_ignored_source` (synthetic source),
//! - DLS on parent/child joins and their `inner_hits` (with and without aggregations),
//! - single-document DLS bound to the document actually returned (duplicate
//!   routed ids on a multi-shard index, GET and mget),
//! - a pipeline rerouting a document (`reroute`, `_index`) re-authorizes the destination.

use base64::Engine;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::security::{hash::Hasher, SecConfig};
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;

const PW: &str = "dls-fls-paths-password";

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

fn auth(user: &str) -> String {
    format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{user}:{PW}")))
}

impl Es {
    async fn call(&self, who: &str, m: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.client.request(m, format!("{}{}", self.base, path)).header("authorization", auth(who));
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b.to_string());
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let text = resp.text().await.unwrap();
        (st, if text.is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap_or(Value::String(text)) })
    }
    async fn ndjson(&self, who: &str, path: &str, lines: &[Value]) -> (u16, Value) {
        let body: String = lines.iter().map(|l| format!("{l}\n")).collect();
        let resp = self.client.post(format!("{}{}", self.base, path)).header("authorization", auth(who)).header("content-type", "application/x-ndjson").body(body).send().await.unwrap();
        let st = resp.status().as_u16();
        (st, resp.json().await.unwrap_or(Value::Null))
    }
    async fn get(&self, w: &str, p: &str) -> (u16, Value) {
        self.call(w, Method::GET, p, None).await
    }
    async fn post(&self, w: &str, p: &str, b: Value) -> (u16, Value) {
        self.call(w, Method::POST, p, Some(b)).await
    }
    async fn put(&self, w: &str, p: &str, b: Value) -> (u16, Value) {
        self.call(w, Method::PUT, p, Some(b)).await
    }
    async fn ok(&self, m: Method, p: &str, b: Option<Value>) -> Value {
        let (st, r) = self.call("admin", m, p, b).await;
        assert!(st == 200 || st == 201, "{p}: {st} {r}");
        r
    }
    async fn user(&self, name: &str, role: &str, body: Value) {
        self.ok(Method::PUT, &format!("/_security/role/{role}"), Some(body)).await;
        self.ok(Method::PUT, &format!("/_security/user/{name}"), Some(json!({"password": PW, "roles": [role]}))).await;
    }
}

fn ids(v: &Value) -> Vec<String> {
    let mut out: Vec<String> = v["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{v}")).iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
    out.sort();
    out
}

/// Every decoded `_ignored_source` value of the hits, as lossy text.
fn decoded_ignored(r: &Value) -> Vec<String> {
    let mut out = Vec::new();
    for h in r["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{r}")) {
        let vals = h.get("_ignored_source").or_else(|| h["fields"].get("_ignored_source"));
        for v in vals.and_then(Value::as_array).into_iter().flatten() {
            let b = base64::engine::general_purpose::STANDARD.decode(v.as_str().unwrap()).unwrap();
            out.push(String::from_utf8_lossy(&b).into_owned());
        }
    }
    out
}

/// #90: `_ignored_source` entries carry original field names and values;
/// a reader restricted by FLS must not recover hidden ones through
/// `stored_fields` / `fields`, while permitted unmapped fields stay readable.
#[tokio::test(flavor = "multi_thread")]
async fn fls_filters_ignored_source() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    es.ok(
        Method::PUT,
        "/syn",
        Some(json!({"settings": {"index.mapping.source.mode": "synthetic", "number_of_shards": 1}, "mappings": {"dynamic": false, "properties": {"public": {"type": "keyword"}}}})),
    )
    .await;
    let doc = json!({
        "public": "pub-value",
        "pub_extra": "visible-extra",
        "secret": {"code": "SENTINEL-OBJ", "n": 7},
        "secret_str": "SENTINEL-STR",
        "mixed": {"open": "mixed-open", "closed": "SENTINEL-MIXED"}
    });
    es.ok(Method::PUT, "/syn/_doc/1?refresh=true", Some(doc)).await;
    es.user(
        "flsreader",
        "fls_public",
        json!({"indices": [{"names": ["syn"], "privileges": ["read"], "field_security": {"grant": ["public", "pub_extra", "mixed.open"]}}]}),
    )
    .await;
    for body in [
        json!({"_source": false, "stored_fields": ["_ignored_source"]}),
        json!({"_source": false, "fields": ["_ignored_source"]}),
        json!({"stored_fields": ["_ignored_source", "_source"]}),
    ] {
        // The superuser sees every encoded entry (the scenario is real).
        let (st, r) = es.post("admin", "/syn/_search", body.clone()).await;
        assert_eq!(st, 200, "{r}");
        let all = decoded_ignored(&r).join("\n");
        assert!(all.contains("SENTINEL-OBJ") && all.contains("SENTINEL-STR") && all.contains("SENTINEL-MIXED"), "admin {body}: {all:?} {r}");
        // The restricted reader sees none of the hidden values...
        let (st, r) = es.post("flsreader", "/syn/_search", body.clone()).await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(ids(&r), ["1"], "{r}");
        let seen = decoded_ignored(&r).join("\n");
        assert!(!seen.contains("SENTINEL"), "reader {body}: hidden value leaked: {seen:?} {r}");
        assert!(!r.to_string().contains("SENTINEL"), "reader {body}: {r}");
        // ... and keeps the permitted ones.
        assert!(seen.contains("visible-extra"), "reader {body}: permitted entry lost: {seen:?} {r}");
        assert!(seen.contains("mixed-open"), "reader {body}: permitted part of an object lost: {seen:?} {r}");
    }
}

/// #101: parent/child joins and their inner hits only see documents the
/// caller's DLS allows, whether or not the request has aggregations.
#[tokio::test(flavor = "multi_thread")]
async fn dls_applies_to_join_inner_hits() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    es.ok(
        Method::PUT,
        "/fam",
        Some(json!({"settings": {"number_of_shards": 1}, "mappings": {"properties": {
            "rel": {"type": "join", "relations": {"p": "c"}},
            "tenant": {"type": "keyword"},
            "body": {"type": "keyword"}
        }}})),
    )
    .await;
    let docs = [
        ("p1", None, json!({"rel": "p", "tenant": "a", "body": "parent-visible"})),
        ("p2", None, json!({"rel": "p", "tenant": "b", "body": "SECRET-PARENT"})),
        ("c1", Some("p1"), json!({"rel": {"name": "c", "parent": "p1"}, "tenant": "b", "body": "SECRET-CHILD"})),
        ("c2", Some("p1"), json!({"rel": {"name": "c", "parent": "p1"}, "tenant": "a", "body": "child-visible"})),
        ("c3", Some("p2"), json!({"rel": {"name": "c", "parent": "p2"}, "tenant": "a", "body": "orphan-visible"})),
    ];
    let mut lines = Vec::new();
    for (id, routing, d) in docs {
        let mut meta = json!({"_index": "fam", "_id": id});
        if let Some(r) = routing {
            meta["routing"] = json!(r);
        }
        lines.push(json!({"index": meta}));
        lines.push(d);
    }
    let (st, r) = es.ndjson("admin", "/_bulk?refresh=true", &lines).await;
    assert_eq!((st, &r["errors"]), (200, &json!(false)), "{r}");
    es.user("tenant_a", "tenant_a", json!({"indices": [{"names": ["fam"], "privileges": ["read"], "query": {"term": {"tenant": "a"}}}]})).await;

    // Sanity: the superuser sees the hidden child as an inner hit.
    let (_, r) = es.post("admin", "/fam/_search", json!({"query": {"has_child": {"type": "c", "query": {"match_all": {}}, "inner_hits": {}}}})).await;
    assert!(r.to_string().contains("SECRET-CHILD"), "{r}");

    for aggs in [None, Some(json!({"t": {"terms": {"field": "tenant"}}}))] {
        let with = |mut q: Value| {
            if let Some(a) = &aggs {
                q["aggs"] = a.clone();
            }
            q
        };
        // Visible parent, hidden child: the child is no inner hit.
        let (st, r) = es.post("tenant_a", "/fam/_search", with(json!({"query": {"has_child": {"type": "c", "query": {"match_all": {}}, "inner_hits": {}}}}))).await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(ids(&r), ["p1"], "aggs={aggs:?}: {r}");
        let inner: Vec<&str> = r["hits"]["hits"][0]["inner_hits"]["c"]["hits"]["hits"].as_array().unwrap_or_else(|| panic!("{r}")).iter().map(|h| h["_id"].as_str().unwrap()).collect();
        assert_eq!(inner, ["c2"], "aggs={aggs:?}: {r}");
        assert!(!r.to_string().contains("SECRET"), "aggs={aggs:?}: {r}");
        // A hidden child cannot make its visible parent match.
        let (st, r) = es.post("tenant_a", "/fam/_search", with(json!({"query": {"has_child": {"type": "c", "query": {"term": {"body": "SECRET-CHILD"}}, "inner_hits": {}}}}))).await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(ids(&r), Vec::<String>::new(), "aggs={aggs:?}: {r}");
        // Visible child, hidden parent: no match through the parent, no parent inner hit.
        let (st, r) = es.post("tenant_a", "/fam/_search", with(json!({"query": {"has_parent": {"parent_type": "p", "query": {"match_all": {}}, "inner_hits": {}}}}))).await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(ids(&r), ["c2"], "aggs={aggs:?}: {r}");
        assert_eq!(r["hits"]["hits"][0]["inner_hits"]["p"]["hits"]["hits"][0]["_id"], "p1", "aggs={aggs:?}: {r}");
        assert!(!r.to_string().contains("SECRET"), "aggs={aggs:?}: {r}");
        // Inner hits of a query that matches the visible child through a
        // `bool` still leave the hidden parent out.
        let (st, r) = es
            .post("tenant_a", "/fam/_search", with(json!({"query": {"bool": {"should": [{"has_parent": {"parent_type": "p", "query": {"match_all": {}}, "inner_hits": {}}}, {"ids": {"values": ["c3"]}}]}}})))
            .await;
        assert_eq!(st, 200, "{r}");
        assert_eq!(ids(&r), ["c2", "c3"], "aggs={aggs:?}: {r}");
        assert!(!r.to_string().contains("SECRET"), "aggs={aggs:?}: {r}");
    }
}

/// #102: a visible copy of an id on one shard cannot authorize the hidden
/// copy of the same id, routed to another shard (GET, GET _source, mget).
#[tokio::test(flavor = "multi_thread")]
async fn single_doc_dls_binds_to_the_returned_shard_copy() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    es.ok(Method::PUT, "/dup", Some(json!({"settings": {"number_of_shards": 4}, "mappings": {"properties": {"dept": {"type": "keyword"}, "secret": {"type": "keyword"}}}}))).await;
    es.ok(Method::PUT, "/dup/_doc/x?routing=k0&refresh=true", Some(json!({"dept": "eng", "secret": "visible-copy"}))).await;
    // A routing that selects another shard: GET there does not find `x`.
    let mut other = None;
    for i in 1..64 {
        let r = format!("k{i}");
        let (st, _) = es.get("admin", &format!("/dup/_doc/x?routing={r}")).await;
        if st == 404 {
            other = Some(r);
            break;
        }
    }
    let other = other.expect("a routing to another shard");
    es.ok(Method::PUT, &format!("/dup/_doc/x?routing={other}&refresh=true"), Some(json!({"dept": "sales", "secret": "SECRET-COPY"}))).await;
    es.user("engdup", "eng_dup", json!({"indices": [{"names": ["dup"], "privileges": ["read"], "query": {"term": {"dept": "eng"}}}]})).await;

    let (st, r) = es.get("engdup", "/dup/_doc/x?routing=k0").await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["_source"]["secret"], "visible-copy", "{r}");
    for realtime in ["true", "false"] {
        let (st, r) = es.get("engdup", &format!("/dup/_doc/x?routing={other}&realtime={realtime}")).await;
        assert_eq!(st, 404, "realtime={realtime}: hidden copy returned: {r}");
        assert!(!r.to_string().contains("SECRET"), "{r}");
        let (st, r) = es.get("engdup", &format!("/dup/_source/x?routing={other}&realtime={realtime}")).await;
        assert_eq!(st, 404, "realtime={realtime}: hidden copy returned: {r}");
    }
    let (st, r) = es.post("engdup", "/_mget", json!({"docs": [{"_index": "dup", "_id": "x", "routing": "k0"}, {"_index": "dup", "_id": "x", "routing": other}]})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["docs"][0]["found"], true, "{r}");
    assert_eq!(r["docs"][1]["found"], false, "hidden copy returned: {r}");
    assert!(!r.to_string().contains("SECRET"), "{r}");
    let (st, r) = es.post("engdup", &format!("/dup/_mget?routing={other}"), json!({"ids": ["x"]})).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["docs"][0]["found"], false, "hidden copy returned: {r}");
}

/// #105: a writer permitted only on the requested index cannot reach another
/// index through a pipeline that reroutes the document (standalone engine
/// ingest, including versioned/conditional writes and bulk); a destination it
/// may write stays reachable.
#[tokio::test(flavor = "multi_thread")]
async fn ingest_reroute_reauthorizes_destination() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    let mapping = json!({"mappings": {"properties": {"f": {"type": "keyword"}}}});
    for ix in ["src", "forbidden", "allowed2"] {
        es.ok(Method::PUT, &format!("/{ix}"), Some(mapping.clone())).await;
    }
    es.ok(Method::PUT, "/_ingest/pipeline/to_forbidden", Some(json!({"processors": [{"reroute": {"destination": "forbidden"}}]}))).await;
    es.ok(Method::PUT, "/_ingest/pipeline/set_forbidden", Some(json!({"processors": [{"set": {"field": "_index", "value": "forbidden"}}]}))).await;
    es.ok(Method::PUT, "/_ingest/pipeline/to_allowed", Some(json!({"processors": [{"reroute": {"destination": "allowed2"}}]}))).await;
    es.user("router", "router", json!({"indices": [{"names": ["src", "allowed2"], "privileges": ["write"]}]})).await;

    let mut n = 0;
    for pipeline in ["to_forbidden", "set_forbidden"] {
        for extra in ["", "&version=5&version_type=external", "&op_type=create"] {
            n += 1;
            let (st, r) = es.put("router", &format!("/src/_doc/d{n}?pipeline={pipeline}{extra}"), json!({"f": "x"})).await;
            assert_eq!(st, 403, "{pipeline}{extra}: rerouted write accepted: {r}");
            let (st, r) = es.post("router", &format!("/src/_doc?pipeline={pipeline}{extra}"), json!({"f": "y"})).await;
            if !extra.contains("version") {
                assert_eq!(st, 403, "{pipeline}{extra} auto id: rerouted write accepted: {r}");
            }
        }
        // An existing document's sequence number as a write condition.
        let (_, r) = es.put("admin", "/src/_doc/cond", json!({"f": "base"})).await;
        let (seq, term) = (r["_seq_no"].clone(), r["_primary_term"].clone());
        let (st, r) = es.put("router", &format!("/src/_doc/cond?pipeline={pipeline}&if_seq_no={seq}&if_primary_term={term}"), json!({"f": "z"})).await;
        assert_eq!(st, 403, "{pipeline} conditional: rerouted write accepted: {r}");
        let (st, r) = es
            .ndjson("router", "/_bulk", &[json!({"index": {"_index": "src", "_id": format!("b-{pipeline}"), "pipeline": pipeline}}), json!({"f": "bulk"}), json!({"create": {"_index": "src", "pipeline": pipeline}}), json!({"f": "bulk2"})])
            .await;
        assert_eq!(st, 200, "{r}");
        for item in r["items"].as_array().unwrap() {
            let (_, v) = item.as_object().unwrap().iter().next().unwrap();
            assert_eq!(v["status"], 403, "{pipeline} bulk: rerouted item accepted: {r}");
        }
    }
    // Permitted destination: accepted, the document lands there.
    let (st, r) = es.put("router", "/src/_doc/ok?pipeline=to_allowed&refresh=true", json!({"f": "fine"})).await;
    assert!(st == 200 || st == 201, "{r}");
    assert_eq!(r["_index"], "allowed2", "{r}");
    let (st, r) = es.ndjson("router", "/_bulk?refresh=true", &[json!({"index": {"_index": "src", "_id": "ok2", "pipeline": "to_allowed"}}), json!({"f": "fine"})]).await;
    assert_eq!((st, &r["errors"]), (200, &json!(false)), "{r}");

    es.ok(Method::POST, "/_refresh", None).await;
    let r = es.ok(Method::GET, "/forbidden/_count", None).await;
    assert_eq!(r["count"], 0, "forbidden destination modified: {r}");
    let r = es.ok(Method::GET, "/allowed2/_count", None).await;
    assert_eq!(r["count"], 2, "{r}");
}
