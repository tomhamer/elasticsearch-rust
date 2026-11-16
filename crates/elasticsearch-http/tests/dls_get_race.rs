//! #102: single-document DLS judges the version it returns. A GET that
//! fetched a hidden version must not return it after a concurrent
//! replacement makes the id visible between the fetch and the check (the
//! check is barriered with `set_dls_check_hook`). Its own test binary: the
//! hook is process-wide.

use base64::Engine;
use elasticsearch_engine::index::WriteReq;
use elasticsearch_engine::node::Node;
use elasticsearch_engine::security::{hash::Hasher, SecConfig};
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const PW: &str = "dls-race-password";

fn auth(user: &str) -> String {
    format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(format!("{user}:{PW}")))
}

#[tokio::test(flavor = "multi_thread")]
async fn get_never_returns_a_hidden_version_replaced_during_the_check() {
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    let mut cfg = SecConfig::default();
    cfg.enabled = true;
    cfg.trial = true;
    cfg.file_users.insert("admin".into(), Hasher::Pbkdf2(1000).hash(PW));
    cfg.file_user_roles.insert("admin".into(), vec!["superuser".into()]);
    node.security.configure(cfg);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let n = node.clone();
    let _task = tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(n, listener).await;
    });
    let client = Client::builder().timeout(std::time::Duration::from_secs(30)).build().unwrap();
    let call = |user: &'static str, method: reqwest::Method, path: String, body: Option<Value>| {
        let client = client.clone();
        let base = base.clone();
        async move {
            let mut r = client.request(method, format!("{base}{path}")).header("authorization", auth(user));
            if let Some(b) = body {
                r = r.header("content-type", "application/json").body(b.to_string());
            }
            let resp = r.send().await.unwrap();
            let st = resp.status().as_u16();
            let text = resp.text().await.unwrap();
            (st, serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text)))
        }
    };
    use reqwest::Method;
    let (st, r) = call("admin", Method::PUT, "/race".into(), Some(json!({"settings": {"number_of_shards": 1}, "mappings": {"properties": {"dept": {"type": "keyword"}, "secret": {"type": "keyword"}}}}))).await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = call("admin", Method::PUT, "/_security/role/eng_race".into(), Some(json!({"indices": [{"names": ["race"], "privileges": ["read"], "query": {"term": {"dept": "eng"}}}]}))).await;
    assert_eq!(st, 200, "{r}");
    let (st, r) = call("admin", Method::PUT, "/_security/user/racer".into(), Some(json!({"password": PW, "roles": ["eng_race"]}))).await;
    assert_eq!(st, 200, "{r}");

    for (round, realtime) in [(0, "true"), (1, "false")] {
        let id = format!("doc{round}");
        let (st, r) = call("admin", Method::PUT, format!("/race/_doc/{id}?refresh=true"), Some(json!({"dept": "sales", "secret": "HIDDEN-OLD"}))).await;
        assert!(st == 200 || st == 201, "{r}");
        // Between the fetch of the hidden version and its DLS check, a
        // replacement makes the id visible (once).
        let fired = Arc::new(AtomicUsize::new(0));
        let (svc, f, hook_id) = (node.write_index("race", false).unwrap(), fired.clone(), id.clone());
        elasticsearch_http::set_dls_check_hook(Some(Arc::new(move || {
            if f.fetch_add(1, Ordering::SeqCst) == 0 {
                let w = svc.write(vec![WriteReq::Index { id: Some(hook_id.clone()), source: json!({"dept": "eng", "secret": "visible-new"}).to_string(), routing: None, cond: Default::default() }], true);
                for res in w {
                    res.unwrap();
                }
                svc.refresh().unwrap();
            }
        })));
        let (st, r) = call("racer", Method::GET, format!("/race/_doc/{id}?realtime={realtime}"), None).await;
        elasticsearch_http::set_dls_check_hook(None);
        assert!(fired.load(Ordering::SeqCst) >= 1, "the barrier never ran");
        assert!(!r.to_string().contains("HIDDEN-OLD"), "realtime={realtime}: hidden version returned: {st} {r}");
        assert!(st == 404 || r["_source"]["dept"] == "eng", "realtime={realtime}: {st} {r}");
        // Same through mget.
        let id2 = format!("{id}-m");
        let (st, r) = call("admin", Method::PUT, format!("/race/_doc/{id2}?refresh=true"), Some(json!({"dept": "sales", "secret": "HIDDEN-OLD"}))).await;
        assert!(st == 200 || st == 201, "{r}");
        let fired = Arc::new(AtomicUsize::new(0));
        let (svc, f, hook_id) = (node.write_index("race", false).unwrap(), fired.clone(), id2.clone());
        elasticsearch_http::set_dls_check_hook(Some(Arc::new(move || {
            if f.fetch_add(1, Ordering::SeqCst) == 0 {
                for res in svc.write(vec![WriteReq::Index { id: Some(hook_id.clone()), source: json!({"dept": "eng", "secret": "visible-new"}).to_string(), routing: None, cond: Default::default() }], true) {
                    res.unwrap();
                }
                svc.refresh().unwrap();
            }
        })));
        let (st, r) = call("racer", Method::POST, format!("/race/_mget?realtime={realtime}"), Some(json!({"ids": [id2]}))).await;
        elasticsearch_http::set_dls_check_hook(None);
        assert_eq!(st, 200, "{r}");
        assert!(fired.load(Ordering::SeqCst) >= 1, "the barrier never ran");
        assert!(!r.to_string().contains("HIDDEN-OLD"), "mget realtime={realtime}: hidden version returned: {r}");
    }
}
