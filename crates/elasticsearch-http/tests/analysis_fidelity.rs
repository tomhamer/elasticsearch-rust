//! Analysis findings closed end to end over HTTP:
//! 299 (`_analyze` limits contain the work), 303 (`keep_words_path`),
//! 304 (pattern tokenizer capture groups), 306 (named filter parameters).

use elasticsearch_engine::node::Node;
use reqwest::{Client, Method};
use serde_json::{json, Value};

struct Es {
    base: String,
    c: Client,
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

    async fn hits(&self, index: &str, q: Value) -> Vec<String> {
        let (st, v) = self.call(Method::POST, &format!("/{index}/_search"), Some(json!({"query": q}))).await;
        assert_eq!(st, 200, "{v}");
        let mut ids: Vec<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
        ids.sort();
        ids
    }
}

async fn start(tmp: &std::path::Path) -> Es {
    let node = Node::open(tmp).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = elasticsearch_http::serve_listener(node, listener).await;
    });
    Es { base: format!("http://{addr}"), c: Client::new() }
}

#[tokio::test]
async fn analysis_settings_take_effect() {
    let conf = tempfile::tempdir().unwrap();
    std::fs::write(conf.path().join("keep.txt"), "# animals\nfox\n\n").unwrap();
    // SAFETY: this is the only test in this binary; no other thread reads the env.
    unsafe { std::env::set_var("ES_PATH_CONF", conf.path()) };
    let tmp = tempfile::tempdir().unwrap();
    let es = start(tmp.path()).await;

    // 299: the token limit stops an expanding chain; offsets are UTF-16.
    let (st, v) = es.call(Method::PUT, "/lim", Some(json!({"settings": {"index.analyze.max_token_count": 3, "analysis": {
        "filter": {"grams": {"type": "ngram", "min_gram": 1, "max_gram": 2}},
        "analyzer": {"grams": {"tokenizer": "standard", "filter": ["grams"]}}}}}))).await;
    assert_eq!(st, 200, "{v}");
    let long = "abcdefghij ".repeat(20_000);
    let (st, v) = es.call(Method::POST, "/lim/_analyze", Some(json!({"analyzer": "grams", "text": long}))).await;
    assert_eq!(st, 400, "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().contains("exceeded the allowed maximum of [3]"), "{v}");
    let (st, v) = es.call(Method::POST, "/lim/_analyze", Some(json!({"analyzer": "standard", "text": "😀 ab cd"}))).await;
    assert_eq!(st, 200, "{v}");
    // `😀` is two UTF-16 units.
    assert_eq!((v["tokens"][1]["token"].clone(), v["tokens"][1]["start_offset"].clone(), v["tokens"][1]["end_offset"].clone()), (json!("ab"), json!(3), json!(5)), "{v}");

    // 303 / 304 / 306 in one index.
    let (st, v) = es.call(Method::PUT, "/docs", Some(json!({"settings": {"analysis": {
        "tokenizer": {"account": {"type": "pattern", "pattern": "([^@]+)@(.+)", "group": 1}},
        "filter": {
            "keepfile": {"type": "keep", "keep_words_path": "keep.txt"},
            "five_hundred": {"type": "limit", "max_token_count": 500},
            "fp": {"type": "fingerprint", "max_output_size": 1024}
        },
        "analyzer": {
            "keep": {"tokenizer": "standard", "filter": ["lowercase", "keepfile"]},
            "account": {"tokenizer": "account"},
            "lim": {"tokenizer": "standard", "filter": ["five_hundred"]},
            "fp": {"tokenizer": "standard", "filter": ["fp"]}
        }}},
        "mappings": {"properties": {
            "k": {"type": "text", "analyzer": "keep", "search_analyzer": "standard"},
            "mail": {"type": "text", "analyzer": "account", "search_analyzer": "keyword"},
            "l": {"type": "text", "analyzer": "lim"},
            "f": {"type": "text", "analyzer": "fp", "search_analyzer": "keyword"}
        }}}))).await;
    assert_eq!(st, 200, "{v}");
    let words: Vec<String> = (0..60).map(|i| format!("w{i:03}")).collect();
    let fp_text = words.join(" ");
    let (st, v) = es
        .call(Method::PUT, "/docs/_doc/1?refresh=true", Some(json!({"k": "Fox dog", "mail": "alice@example.org", "l": "alpha beta gamma", "f": fp_text})))
        .await;
    assert_eq!(st, 201, "{v}");
    assert_eq!(es.hits("docs", json!({"match": {"k": "fox"}})).await, vec!["1"]);
    assert!(es.hits("docs", json!({"match": {"k": "dog"}})).await.is_empty());
    assert_eq!(es.hits("docs", json!({"term": {"mail": "alice"}})).await, vec!["1"]);
    assert!(es.hits("docs", json!({"term": {"mail": "alice@example.org"}})).await.is_empty());
    assert_eq!(es.hits("docs", json!({"match": {"l": "gamma"}})).await, vec!["1"]);
    // A 299-character fingerprint survives a configured 1024 limit.
    assert_eq!(es.hits("docs", json!({"term": {"f": fp_text}})).await, vec!["1"]);

    // A missing word list fails the index instead of indexing nothing.
    let (st, v) = es.call(Method::PUT, "/bad", Some(json!({"settings": {"analysis": {
        "filter": {"k": {"type": "keep", "keep_words_path": "missing.txt"}},
        "analyzer": {"a": {"tokenizer": "standard", "filter": ["k"]}}}}}))).await;
    assert_eq!(st, 400, "{v}");
    assert!(v.to_string().contains("IOException while reading keep_words_path"), "{v}");
}
