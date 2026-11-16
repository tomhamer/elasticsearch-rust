//! Closure tests for the c33 security review: derived API keys hold no
//! privileges, an API key owns only itself, run-as never reaches a
//! disabled user, a refreshed token keeps the original authentication, and
//! a `lookup` runtime field needs read on its target index.

use base64::Engine;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::security::{hash::Hasher, SecConfig};
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;

const PW: &str = "c33-closure-test-password";

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
    cfg.token_enabled = true;
    cfg.trial = true;
    cfg.password_hashing = Hasher::Pbkdf2(1000);
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
    Bearer(String),
}

fn auth(a: &As) -> String {
    match a {
        As::User(u) => format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{u}:{PW}"))),
        As::Key(k) => format!("ApiKey {k}"),
        As::Bearer(t) => format!("Bearer {t}"),
    }
}

const ADMIN: As = As::User("admin");

impl Es {
    async fn call_with(&self, who: &As, run_as: Option<&str>, m: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.client.request(m, format!("{}{}", self.base, path)).header("authorization", auth(who));
        if let Some(target) = run_as {
            r = r.header("es-security-runas-user", target);
        }
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b.to_string());
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let text = resp.text().await.unwrap();
        (st, if text.is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap_or(Value::String(text)) })
    }
    async fn call(&self, who: &As, m: Method, path: &str, body: Option<Value>) -> (u16, Value) {
        self.call_with(who, None, m, path, body).await
    }
}

async fn put_role(es: &Es, name: &str, body: Value) {
    let (st, r) = es.call(&ADMIN, Method::PUT, &format!("/_security/role/{name}"), Some(body)).await;
    assert_eq!(st, 200, "role {name}: {r}");
}

async fn put_user(es: &Es, name: &str, roles: Value) {
    let (st, r) = es.call(&ADMIN, Method::PUT, &format!("/_security/user/{name}"), Some(json!({"password": PW, "roles": roles}))).await;
    assert_eq!(st, 200, "user {name}: {r}");
}

/// (id, encoded) of a new key created by `owner`.
async fn create_key(es: &Es, owner: &As, body: Value) -> (String, String) {
    let (st, r) = es.call(owner, Method::POST, "/_security/api_key", Some(body)).await;
    assert_eq!(st, 200, "create key: {r}");
    (r["id"].as_str().unwrap().to_string(), r["encoded"].as_str().unwrap().to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn derived_api_keys_hold_no_privileges() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    put_role(&es, "key_owner", json!({"cluster": ["manage_own_api_key", "monitor"]})).await;
    put_user(&es, "owner", json!(["key_owner"])).await;
    put_user(&es, "boss", json!(["superuser"])).await;
    // A parent key asking for more than its owner holds.
    let (_, parent) = create_key(&es, &As::User("owner"), json!({"name": "parent", "role_descriptors": {"r": {
        "cluster": ["all"], "run_as": ["*"], "indices": [{"names": ["*"], "privileges": ["all"]}]
    }}})).await;
    let parent = As::Key(parent);
    // A derived key must have empty descriptors: `run_as` counts too.
    let (st, r) = es.call(&parent, Method::POST, "/_security/api_key", Some(json!({"name": "derived", "role_descriptors": {"d": {"run_as": ["boss"]}}}))).await;
    assert_eq!(st, 400, "a derived key was created with run_as: {r}");
    assert_eq!(r["error"]["type"], "illegal_argument_exception", "{r}");
    // An empty derived key authenticates but holds nothing.
    let (_, derived) = create_key(&es, &parent, json!({"name": "derived", "role_descriptors": {"d": {}}})).await;
    let derived = As::Key(derived);
    assert_eq!(es.call(&derived, Method::GET, "/_security/_authenticate", None).await.0, 200);
    let (st, r) = es.call(&derived, Method::GET, "/_cluster/health", None).await;
    assert_eq!(st, 403, "a derived key holds privileges: {r}");
    let (st, r) = es.call_with(&derived, Some("boss"), Method::GET, "/_security/_authenticate", None).await;
    assert_eq!(st, 403, "a derived key ran as another user: {r}");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_api_key_owns_only_itself() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    put_role(&es, "key_owner", json!({"cluster": ["manage_own_api_key"]})).await;
    put_user(&es, "owner", json!(["key_owner"])).await;
    let owner = As::User("owner");
    let (a_id, a) = create_key(&es, &owner, json!({"name": "a"})).await;
    let (b_id, _) = create_key(&es, &owner, json!({"name": "b"})).await;
    let a = As::Key(a);
    // Key A cannot read its sibling B...
    let (_, r) = es.call(&a, Method::GET, &format!("/_security/api_key?id={b_id}"), None).await;
    assert!(!r.to_string().contains(&b_id), "an API key read another key of its owner: {r}");
    let (_, r) = es.call(&a, Method::GET, "/_security/api_key?owner=true", None).await;
    assert!(!r.to_string().contains(&b_id), "owner=true listed a sibling key: {r}");
    // ...nor invalidate it.
    let (st, r) = es.call(&a, Method::DELETE, "/_security/api_key", Some(json!({"ids": [b_id.clone()]}))).await;
    assert_eq!(st, 403, "an API key invalidated another key of its owner: {r}");
    let (st, r) = es.call(&a, Method::DELETE, "/_security/api_key", Some(json!({"owner": true}))).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["invalidated_api_keys"], json!([a_id]), "owner=true reached beyond the key itself: {r}");
    // The owner still sees B active.
    let (st, r) = es.call(&owner, Method::GET, &format!("/_security/api_key?id={b_id}"), None).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(r["api_keys"][0]["invalidated"], false, "{r}");
}

#[tokio::test(flavor = "multi_thread")]
async fn run_as_a_disabled_user_fails_authentication() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    put_role(&es, "runner", json!({"run_as": ["target"]})).await;
    put_user(&es, "runner", json!(["runner"])).await;
    put_user(&es, "target", json!(["superuser"])).await;
    let runner = As::User("runner");
    let (st, r) = es.call_with(&runner, Some("target"), Method::GET, "/_security/_authenticate", None).await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = es.call(&ADMIN, Method::PUT, "/_security/user/target/_disable", None).await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = es.call_with(&runner, Some("target"), Method::GET, "/_security/_authenticate", None).await;
    assert_eq!(st, 401, "run-as reached a disabled user: {r}");
    assert_eq!(r["error"]["type"], "security_exception", "{r}");
    let (st, _) = es.call_with(&runner, Some("target"), Method::GET, "/_security/user", None).await;
    assert_eq!(st, 401);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_refreshed_token_keeps_the_original_authentication() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    put_role(&es, "plain", json!({"cluster": ["monitor"]})).await;
    put_user(&es, "plain", json!(["plain"])).await;
    let (st, token) = es.call(&ADMIN, Method::POST, "/_security/oauth2/token", Some(json!({"grant_type": "password", "username": "plain", "password": PW}))).await;
    assert_eq!(st, 200, "{token}");
    let refresh = token["refresh_token"].as_str().unwrap().to_string();
    // Only the creating client may refresh (ES `checkClientCanRefresh`).
    put_role(&es, "token_client", json!({"cluster": ["manage_token"]})).await;
    put_user(&es, "otherclient", json!(["token_client"])).await;
    let (st, r) = es.call(&As::User("otherclient"), Method::POST, "/_security/oauth2/token", Some(json!({"grant_type": "refresh_token", "refresh_token": refresh.clone()}))).await;
    assert_eq!(st, 400, "another client refreshed the token: {r}");
    assert_eq!(r["error"]["error_description"], "tokens must be refreshed by the creating client", "{r}");
    // The user goes away; the refreshing client is a superuser.
    let (st, r) = es.call(&ADMIN, Method::DELETE, "/_security/user/plain", None).await;
    assert_eq!(st, 200, "{r}");
    let (st, again) = es.call(&ADMIN, Method::POST, "/_security/oauth2/token", Some(json!({"grant_type": "refresh_token", "refresh_token": refresh.clone()}))).await;
    assert_eq!(st, 200, "{again}");
    let bearer = As::Bearer(again["access_token"].as_str().unwrap().to_string());
    let (st, me) = es.call(&bearer, Method::GET, "/_security/_authenticate", None).await;
    assert_eq!(st, 200, "{me}");
    assert_eq!(me["username"], "plain", "{me}");
    assert_eq!(me["roles"], json!(["plain"]), "the refreshed token took the refreshing client's roles: {me}");
    let (st, r) = es.call(&bearer, Method::GET, "/_security/user", None).await;
    assert_eq!(st, 403, "the refreshed token gained the client's privileges: {r}");
    // A refresh token is used once.
    let (st, r) = es.call(&ADMIN, Method::POST, "/_security/oauth2/token", Some(json!({"grant_type": "refresh_token", "refresh_token": refresh}))).await;
    assert_eq!(st, 400, "{r}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_lookup_runtime_field_needs_read_on_its_target() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    for index in ["open", "secret"] {
        let (st, r) = es.call(&ADMIN, Method::PUT, &format!("/{index}"), Some(json!({"mappings": {"properties": {"k": {"type": "keyword"}, "s": {"type": "keyword"}}}}))).await;
        assert_eq!(st, 200, "{r}");
    }
    let (st, r) = es.call(&ADMIN, Method::PUT, "/open/_doc/1?refresh=true", Some(json!({"k": "x"}))).await;
    assert!(st == 200 || st == 201, "{r}");
    let (st, r) = es.call(&ADMIN, Method::PUT, "/secret/_doc/1?refresh=true", Some(json!({"k": "x", "s": "classified"}))).await;
    assert!(st == 200 || st == 201, "{r}");
    put_role(&es, "reader", json!({"indices": [{"names": ["open"], "privileges": ["read"]}]})).await;
    put_user(&es, "reader", json!(["reader"])).await;
    let body = json!({
        "runtime_mappings": {"l": {"type": "lookup", "target_index": "secret", "input_field": "k", "target_field": "k", "fetch_fields": ["s"]}},
        "fields": ["l"]
    });
    // The admin can read both: the lookup works.
    let (st, r) = es.call(&ADMIN, Method::POST, "/open/_search", Some(body.clone())).await;
    assert_eq!(st, 200, "{r}");
    assert!(r.to_string().contains("classified"), "lookup did not resolve for the admin: {r}");
    let (st, r) = es.call(&As::User("reader"), Method::POST, "/open/_search", Some(body)).await;
    assert!(!r.to_string().contains("classified"), "a lookup field read an index the caller cannot read: {r}");
    assert_eq!(st, 403, "{r}");
}
