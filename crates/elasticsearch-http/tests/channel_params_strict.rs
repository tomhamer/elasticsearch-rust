//! ES `AbstractRestChannel` / `RestController.validateErrorTrace`: `pretty`,
//! `human` and `error_trace` parse strictly (`Booleans.parseBoolean`), for
//! every request, matched route or not.

use elasticsearch_engine::node::Node;
use serde_json::{json, Value};
use std::sync::Arc;

async fn start(dir: &std::path::Path) -> (String, Arc<Node>, tokio::task::JoinHandle<()>) {
    let node = Node::open(dir).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let n = node.clone();
    let task = tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(n, listener).await;
    });
    (format!("http://{addr}"), node, task)
}

async fn get(base: &str, path: &str) -> (u16, String) {
    let r = reqwest::get(format!("{base}{path}")).await.unwrap();
    (r.status().as_u16(), r.text().await.unwrap())
}

fn bad(value: &str) -> String {
    format!("Failed to parse value [{value}] as only [true] or [false] are allowed.")
}

#[tokio::test(flavor = "multi_thread")]
async fn pretty_human_and_error_trace_parse_strictly() {
    let tmp = tempfile::tempdir().unwrap();
    let (base, node, _task) = start(tmp.path()).await;

    // `pretty` / `human`: a bad value is the request's 400, rendered on a
    // channel without the request's parameters (not pretty, no trace).
    for (path, value) in [("/?pretty=yes", "yes"), ("/?pretty=TRUE", "TRUE"), ("/?human=1", "1"), ("/_no_such_api?human=on", "on"), ("/?pretty=no&error_trace=true", "no")] {
        let (status, text) = get(&base, path).await;
        assert_eq!(status, 400, "{path}: {text}");
        assert!(!text.contains('\n'), "{path}: not pretty: {text}");
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["error"]["type"], "illegal_argument_exception", "{path}: {v}");
        assert_eq!(v["error"]["reason"], bad(value), "{path}: {v}");
        assert!(v["error"].get("stack_trace").is_none(), "{path}: {v}");
    }
    // `pretty` is read before `human`.
    let (_, text) = get(&base, "/?human=h&pretty=p").await;
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["error"]["reason"], bad("p"));

    // Present-but-empty is true; whitespace only is the default (false).
    let (status, text) = get(&base, "/?pretty").await;
    assert_eq!(status, 200);
    assert!(text.contains("\n  "), "{text}");
    let (status, text) = get(&base, "/?pretty=%20").await;
    assert_eq!(status, 200);
    assert!(!text.contains('\n'), "{text}");
    assert_eq!(get(&base, "/?human=%20%09").await.0, 200);
    assert_eq!(get(&base, "/?error_trace=%20").await.0, 200);

    // `error_trace`: a simple error response, matched route or not.
    for path in ["/?error_trace=nope", "/_no_such_api?error_trace=nope", "/_search?error_trace=yes"] {
        let (status, text) = get(&base, path).await;
        assert_eq!(status, 400, "{path}: {text}");
        let value = path.rsplit('=').next().unwrap();
        assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), json!({"error": bad(value), "status": 400}), "{path}");
    }

    // With detailed errors disabled, `error_trace=true` is refused for
    // every request, before handler lookup.
    node.node_settings.write().insert("http.detailed_errors.enabled".into(), json!(false));
    for path in ["/?error_trace=true", "/?error_trace", "/_no_such_api?error_trace=true", "/_search?error_trace=true"] {
        let (status, text) = get(&base, path).await;
        assert_eq!(status, 400, "{path}: {text}");
        assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), json!({"error": "error traces in responses are disabled.", "status": 400}), "{path}");
    }
    assert_eq!(get(&base, "/?error_trace=false").await.0, 200);
    assert_eq!(get(&base, "/").await.0, 200);
}
