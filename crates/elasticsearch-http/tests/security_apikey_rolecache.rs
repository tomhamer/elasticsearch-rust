//! API keys stay under their owner's privilege ceiling through run-as and
//! OAuth2 token grants, role changes take effect on the next request, and
//! the security `_clear_cache` APIs really invalidate.

use base64::Engine;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::security::{hash::Hasher, SecConfig};
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;

const PW: &str = "apikey-ceiling-test-password";

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

async fn client_credentials(es: &Es, who: &As, run_as: Option<&str>) -> (u16, Value) {
    es.call_with(who, run_as, Method::POST, "/_security/oauth2/token", Some(json!({"grant_type": "client_credentials"}))).await
}

#[tokio::test(flavor = "multi_thread")]
async fn limited_api_key_cannot_escape_its_owner_through_run_as_or_tokens() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, _node) = start(tmp.path()).await;
    put_role(&es, "key_owner", json!({"cluster": ["manage_own_api_key", "manage_token", "monitor"]})).await;
    put_role(&es, "runs_as_victim", json!({"run_as": ["victim"]})).await;
    put_user(&es, "owner", json!(["key_owner"])).await;
    put_user(&es, "runner", json!(["key_owner", "runs_as_victim"])).await;
    put_user(&es, "victim", json!(["superuser"])).await;
    let owner = As::User("owner");
    let overbroad = json!({"name": "overbroad", "role_descriptors": {"r": {
        "cluster": ["all"], "indices": [{"names": ["*"], "privileges": ["all"]}], "run_as": ["victim"]
    }}});

    let (_, key) = create_key(&es, &owner, overbroad.clone()).await;
    let key = As::Key(key);
    // The owner's ceiling applies to the key itself.
    assert_eq!(es.call(&key, Method::GET, "/_security/user", None).await.0, 403);
    // Run-as: the key asks for it, its owner may not.
    let (st, r) = es.call_with(&key, Some("victim"), Method::GET, "/_security/user", None).await;
    assert_eq!(st, 403, "API key ran as a user its owner cannot: {r}");
    assert!(r["error"]["reason"].as_str().unwrap().contains("unauthorized to run as [victim]"), "{r}");
    let (st, r) = es.call_with(&key, Some("victim"), Method::GET, "/_security/_authenticate", None).await;
    assert_eq!(st, 403, "{r}");
    // ...nor through a client_credentials token.
    assert_eq!(client_credentials(&es, &key, Some("victim")).await.0, 403);

    // A token issued to the key is still the key, still limited.
    let (st, token) = client_credentials(&es, &key, None).await;
    assert_eq!(st, 200, "{token}");
    assert!(token.get("refresh_token").is_none(), "client_credentials never issues a refresh token: {token}");
    assert_eq!(token["authentication"]["authentication_type"], "api_key", "{token}");
    let bearer = As::Bearer(token["access_token"].as_str().unwrap().to_string());
    let (st, me) = es.call(&bearer, Method::GET, "/_security/_authenticate", None).await;
    assert_eq!(st, 200, "{me}");
    assert_eq!(me["username"], "owner", "{me}");
    assert_eq!(me["authentication_type"], "token", "{me}");
    assert_eq!(me["authentication_realm"]["name"], "_es_api_key", "{me}");
    assert!(me["api_key"]["id"].is_string(), "the token dropped the API key subject: {me}");
    for (m, path, body) in [
        (Method::GET, "/_security/user", None),
        (Method::PUT, "/_security/user/mallory", Some(json!({"password": PW, "roles": ["superuser"]}))),
        (Method::PUT, "/escalated", Some(json!({}))),
    ] {
        let (st, r) = es.call(&bearer, m, path, body).await;
        assert_eq!(st, 403, "the token gained a privilege the key's owner lacks ({path}): {r}");
    }
    assert_eq!(es.call(&bearer, Method::GET, "/_cluster/health", None).await.0, 200);
    let (st, r) = es.call_with(&bearer, Some("victim"), Method::GET, "/_security/user", None).await;
    assert_eq!(st, 403, "{r}");
    // Grant API key with the token's access_token and run_as: still bounded.
    let (st, r) = es.call(&ADMIN, Method::POST, "/_security/api_key/grant", Some(json!({
        "grant_type": "access_token", "access_token": token["access_token"], "run_as": "victim", "api_key": {"name": "granted"}
    }))).await;
    assert_eq!(st, 403, "{r}");

    // Positive control: an owner with run_as [victim] and a key allowing it.
    let (runner_key_id, runner_key) = create_key(&es, &As::User("runner"), overbroad).await;
    let runner_key = As::Key(runner_key);
    let (st, me) = es.call_with(&runner_key, Some("victim"), Method::GET, "/_security/_authenticate", None).await;
    assert_eq!(st, 200, "{me}");
    assert_eq!(me["username"], "victim", "{me}");
    let (st, token) = client_credentials(&es, &runner_key, Some("victim")).await;
    assert_eq!(st, 200, "{token}");
    let run_as_bearer = As::Bearer(token["access_token"].as_str().unwrap().to_string());
    let (st, me) = es.call(&run_as_bearer, Method::GET, "/_security/_authenticate", None).await;
    assert_eq!(st, 200, "{me}");
    assert_eq!(me["username"], "victim", "{me}");
    assert_eq!(me["authentication_realm"]["name"], "_es_api_key", "the run-as token lost its API key subject: {me}");
    // Invalidating the key ends its tokens.
    let (st, r) = es.call(&ADMIN, Method::DELETE, "/_security/api_key", Some(json!({"ids": [runner_key_id]}))).await;
    assert_eq!(st, 200, "{r}");
    assert_eq!(es.call(&run_as_bearer, Method::GET, "/_security/_authenticate", None).await.0, 401);

    // Password grant tokens are unaffected.
    let (st, token) = es.call(&key, Method::POST, "/_security/oauth2/token", Some(json!({"grant_type": "password", "username": "victim", "password": PW}))).await;
    assert_eq!(st, 200, "{token}");
    let (st, me) = es.call(&As::Bearer(token["access_token"].as_str().unwrap().to_string()), Method::GET, "/_security/_authenticate", None).await;
    assert_eq!((st, me["username"].as_str()), (200, Some("victim")), "{me}");
}

#[tokio::test(flavor = "multi_thread")]
async fn role_changes_and_clear_cache_apis_invalidate() {
    let tmp = tempfile::tempdir().unwrap();
    let (es, node) = start(tmp.path()).await;
    put_role(&es, "r", json!({"cluster": ["all"]})).await;
    put_user(&es, "u", json!(["r"])).await;
    let u = As::User("u");
    assert_eq!(es.call(&u, Method::GET, "/_security/user", None).await.0, 200);
    // A downgrade applies to the next request.
    put_role(&es, "r", json!({"cluster": ["monitor"]})).await;
    assert_eq!(es.call(&u, Method::GET, "/_security/user", None).await.0, 403);
    assert_eq!(es.call(&u, Method::GET, "/_cluster/health", None).await.0, 200);

    let roles = || node.security.cache_keys().0;
    let verified = || node.security.cache_keys().1;
    let check_nodes = |r: &Value| {
        assert_eq!(r["_nodes"]["successful"], 1, "{r}");
        assert_eq!(r["nodes"].as_object().map(|n| n.len()), Some(1), "{r}");
        assert!(r["cluster_name"].is_string(), "{r}");
    };

    assert!(roles().iter().any(|k| k == "roles:r"));
    let (st, r) = es.call(&ADMIN, Method::POST, "/_security/role/other/_clear_cache", None).await;
    assert_eq!(st, 200, "{r}");
    check_nodes(&r);
    let (st, r) = es.call(&ADMIN, Method::POST, "/_security/role/r/_clear_cache", None).await;
    assert_eq!(st, 200, "{r}");
    assert!(!roles().iter().any(|k| k == "roles:r"), "role clear-cache left the role cached: {:?}", roles());

    assert!(verified().iter().any(|k| k == "basic\u{0}u"));
    let (st, r) = es.call(&ADMIN, Method::POST, "/_security/realm/default_native/_clear_cache?usernames=u", None).await;
    assert_eq!(st, 200, "{r}");
    check_nodes(&r);
    assert!(!verified().iter().any(|k| k == "basic\u{0}u"), "{:?}", verified());

    let (id, key) = create_key(&es, &ADMIN, json!({"name": "cached"})).await;
    let key = As::Key(key);
    assert_eq!(es.call(&key, Method::GET, "/_cluster/health", None).await.0, 200);
    assert!(verified().iter().any(|k| *k == format!("apikey\u{0}{id}")));
    assert!(roles().iter().any(|k| *k == format!("apikey-limit:{id}")));
    let (st, r) = es.call(&ADMIN, Method::POST, &format!("/_security/api_key/{id}/_clear_cache"), None).await;
    assert_eq!(st, 200, "{r}");
    check_nodes(&r);
    assert!(!verified().iter().any(|k| *k == format!("apikey\u{0}{id}")), "{:?}", verified());
    assert!(!roles().iter().any(|k| k.ends_with(&id)), "{:?}", roles());

    assert_eq!(es.call(&u, Method::GET, "/_cluster/health", None).await.0, 200);
    assert!(roles().iter().any(|k| k == "roles:r"));
    let (st, r) = es.call(&ADMIN, Method::POST, "/_security/privilege/myapp/_clear_cache", None).await;
    assert_eq!(st, 200, "{r}");
    assert!(!roles().iter().any(|k| k == "roles:r"), "privileges clear-cache also clears roles (ES)");

    let (st, r) = es.call(&ADMIN, Method::POST, "/_security/service/elastic/fleet-server/credential/token/tok1", None).await;
    assert_eq!(st, 200, "{r}");
    let service = As::Bearer(r["token"]["value"].as_str().unwrap().to_string());
    assert_eq!(es.call(&service, Method::GET, "/_security/_authenticate", None).await.0, 200);
    assert!(verified().iter().any(|k| k == "svc\u{0}elastic/fleet-server/tok1"));
    let (st, r) = es.call(&ADMIN, Method::POST, "/_security/service/elastic/fleet-server/credential/token/tok1/_clear_cache", None).await;
    assert_eq!(st, 200, "{r}");
    check_nodes(&r);
    assert!(!verified().iter().any(|k| k == "svc\u{0}elastic/fleet-server/tok1"), "{:?}", verified());
    // Service accounts cannot create OAuth2 tokens (ES), even with manage_token.
    let (st, r) = es.call(&ADMIN, Method::POST, "/_security/service/elastic/kibana/credential/token/tok2", None).await;
    assert_eq!(st, 200, "{r}");
    let kibana = As::Bearer(r["token"]["value"].as_str().unwrap().to_string());
    let (st, r) = client_credentials(&es, &kibana, None).await;
    assert_eq!(st, 500, "{r}");
    assert!(r["error"]["reason"].as_str().unwrap().contains("not supported for service accounts"), "{r}");

    // Clear-cache needs its cluster privilege.
    assert_eq!(es.call(&u, Method::POST, "/_security/role/r/_clear_cache", None).await.0, 403);
}
