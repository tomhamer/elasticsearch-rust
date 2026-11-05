//! HTTP tests for enrich policies (+ `enrich` processor, ES|QL `ENRICH`),
//! transforms, and the ingest/text-structure extras (geoip, user_agent,
//! attachment, `_text_structure`).

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::Value;

pub struct Es {
    base: String,
    c: Client,
}

pub async fn start(dir: &std::path::Path) -> Es {
    let node = Node::open(dir).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    Es { base: format!("http://{addr}"), c: Client::new() }
}

#[allow(dead_code)]
impl Es {
    pub async fn call(&self, m: Method, p: &str, body: Option<Value>) -> (u16, Value) {
        let mut r = self.c.request(m, format!("{}{}", self.base, p));
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b.to_string());
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        let t = resp.text().await.unwrap();
        (st, if t.is_empty() { Value::Null } else { serde_json::from_str(&t).unwrap_or(Value::String(t)) })
    }
    /// Raw text body (`_cat`, text formats).
    pub async fn text(&self, m: Method, p: &str, body: Option<String>) -> (u16, String) {
        let mut r = self.c.request(m, format!("{}{}", self.base, p));
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b);
        }
        let resp = r.send().await.unwrap();
        let st = resp.status().as_u16();
        (st, resp.text().await.unwrap())
    }
    pub async fn get(&self, p: &str) -> (u16, Value) {
        self.call(Method::GET, p, None).await
    }
    pub async fn post(&self, p: &str, b: Value) -> (u16, Value) {
        self.call(Method::POST, p, Some(b)).await
    }
    pub async fn post_empty(&self, p: &str) -> (u16, Value) {
        self.call(Method::POST, p, None).await
    }
    pub async fn put(&self, p: &str, b: Value) -> (u16, Value) {
        self.call(Method::PUT, p, Some(b)).await
    }
    pub async fn put_empty(&self, p: &str) -> (u16, Value) {
        self.call(Method::PUT, p, None).await
    }
    pub async fn delete(&self, p: &str) -> (u16, Value) {
        self.call(Method::DELETE, p, None).await
    }
    /// NDJSON bulk.
    pub async fn bulk(&self, p: &str, lines: &[Value]) -> (u16, Value) {
        let mut b = String::new();
        for l in lines {
            b.push_str(&l.to_string());
            b.push('\n');
        }
        let resp = self.c.post(format!("{}{}", self.base, p)).header("content-type", "application/x-ndjson").body(b).send().await.unwrap();
        let st = resp.status().as_u16();
        (st, resp.json().await.unwrap())
    }
    pub async fn source(&self, index: &str, id: &str) -> Value {
        let (st, v) = self.get(&format!("/{index}/_doc/{id}")).await;
        assert_eq!(st, 200, "{v}");
        v["_source"].clone()
    }
}

#[path = "enrich_transform/enrich.rs"]
mod enrich;
#[path = "enrich_transform/transform.rs"]
mod transform;
#[path = "enrich_transform/text.rs"]
mod text;
