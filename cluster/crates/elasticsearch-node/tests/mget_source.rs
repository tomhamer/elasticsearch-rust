//! `_mget` body `_source` / `stored_fields` handling across shards on two
//! nodes, as ES 9.5.5's `MultiGetRequest#add` parses it: only `docs` and
//! `ids` are top-level keys (a body-level `_source` is a 400 before any
//! shard is read, not a default), per-doc `_source` (boolean, string,
//! array, includes/excludes object) and `stored_fields` apply to every doc
//! whichever shard and node hold it, and URL parameters are the defaults.

mod common;
use common::start;
use reqwest::Method;
use serde_json::{json, Value};

/// A request with a raw body (exact byte positions, NDJSON).
async fn raw(c: &common::Cluster, node: usize, method: Method, path: &str, body: String, ct: &str) -> (u16, Value) {
    let r = c.c.request(method, format!("http://127.0.0.1:{}{path}", c.p.http[node])).header("content-type", ct).body(body).send().await.unwrap();
    let st = r.status().as_u16();
    (st, r.json().await.unwrap_or(Value::Null))
}

fn docs(v: &Value) -> &Vec<Value> {
    v["docs"].as_array().unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mget_body_source_variants_across_shards() {
    let c = start(&[]).await;
    let (st, body) = c.call(c.master, Method::PUT, "/m", Some(json!({
        "settings": {"number_of_shards": 4, "number_of_replicas": 0},
        "mappings": {"properties": {"a": {"type": "keyword", "store": true}, "b": {"type": "long"}, "o": {"properties": {"x": {"type": "keyword"}, "y": {"type": "keyword"}}}}}
    }))).await;
    assert_eq!(st, 200, "{body}");
    let ids: Vec<String> = (0..12).map(|i| i.to_string()).collect();
    let mut lines = String::new();
    for id in &ids {
        lines.push_str(&format!("{{\"index\":{{\"_index\":\"m\",\"_id\":\"{id}\"}}}}\n{{\"a\":\"v{id}\",\"b\":{id},\"o\":{{\"x\":\"X\",\"y\":\"Y\"}}}}\n"));
    }
    let (st, body) = raw(&c, c.master, Method::POST, "/_bulk?refresh=true", lines, "application/x-ndjson").await;
    assert_eq!(st, 200, "{body}");
    assert_eq!(body["errors"], json!(false), "{body}");
    for node in [c.master, c.other] {
        // Body-level keys other than `docs` / `ids` are parse errors.
        let body = r#"{"ids":["1"],"_source":false}"#;
        let (st, e) = raw(&c, node, Method::POST, "/m/_mget", body.to_string(), "application/json").await;
        assert_eq!(st, 400, "{e}");
        assert_eq!(e["error"]["type"], "parsing_exception", "{e}");
        assert_eq!(e["error"]["reason"], "unexpected token [VALUE_BOOLEAN], expected [FIELD_NAME] or [START_ARRAY]", "{e}");
        assert_eq!((e["error"]["line"].clone(), e["error"]["col"].clone()), (json!(1), json!(24)), "{e}");
        for (extra, reason) in [
            (json!({"_source": ["a"]}), "unknown key [_source] for a START_ARRAY, expected [docs] or [ids]"),
            (json!({"stored_fields": ["a"]}), "unknown key [stored_fields] for a START_ARRAY, expected [docs] or [ids]"),
            (json!({"_source": {"includes": ["a"]}}), "unexpected token [START_OBJECT], expected [FIELD_NAME] or [START_ARRAY]"),
        ] {
            let mut b = json!({"ids": ids});
            b.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            let (st, e) = c.call(node, Method::POST, "/m/_mget", Some(b)).await;
            assert_eq!((st, e["error"]["reason"].as_str()), (400, Some(reason)), "{e}");
        }
        let (st, e) = c.call(node, Method::POST, "/_mget", Some(json!({"docs": [{"_index": "m", "_id": "1"}], "_source": false}))).await;
        assert_eq!(st, 400, "{e}");
        // Per-doc `_source: false`: every doc found, none with a source.
        let (st, v) = c.call(node, Method::POST, "/m/_mget", Some(json!({"docs": ids.iter().map(|i| json!({"_id": i, "_source": false})).collect::<Vec<_>>()}))).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(docs(&v).len(), ids.len());
        for d in docs(&v) {
            assert_eq!(d["found"], json!(true), "{d}");
            assert!(d.get("_source").is_none() && d.get("error").is_none(), "{d}");
        }
        // Per-doc filters.
        let (st, v) = c.call(node, Method::POST, "/m/_mget", Some(json!({"docs": [
            {"_id": "1", "_source": "a"},
            {"_id": "2", "_source": ["a", "o.x"]},
            {"_id": "3", "_source": {"includes": "o.*", "excludes": ["o.y"]}},
            {"_id": "4", "stored_fields": ["a"]},
            {"_id": "5", "stored_fields": "a", "_source": true},
            {"_id": "6", "stored_fields": ["a"], "_source": ["b"]},
            {"_index": "m", "_id": "7"}
        ]}))).await;
        assert_eq!(st, 200, "{v}");
        let d = docs(&v);
        assert_eq!(d[0]["_source"], json!({"a": "v1"}), "{v}");
        assert_eq!(d[1]["_source"], json!({"a": "v2", "o": {"x": "X"}}), "{v}");
        assert_eq!(d[2]["_source"], json!({"o": {"x": "X"}}), "{v}");
        assert!(d[3].get("_source").is_none(), "{v}");
        assert_eq!(d[3]["fields"], json!({"a": ["v4"]}), "{v}");
        assert_eq!((d[4]["_source"]["b"].clone(), d[4]["fields"].clone()), (json!(5), json!({"a": ["v5"]})), "{v}");
        assert_eq!((d[5]["_source"].clone(), d[5]["fields"].clone()), (json!({"b": 6}), json!({"a": ["v6"]})), "{v}");
        assert_eq!(d[6]["_source"]["a"], "v7", "{v}");
        // Unknown and unsupported element fields.
        for (doc, kind, reason) in [
            (json!({"_id": "1", "foo": "x"}), "parse_exception", "failed to parse multi get request. unknown field [foo]"),
            (json!({"_id": "1", "fields": ["a"]}), "parsing_exception", "Unsupported field [fields] used, expected [stored_fields] instead"),
            (json!({"_id": "1", "_source": 5}), "parse_exception", "illegal type for _source: [VALUE_NUMBER]"),
            (json!({"_id": "1", "foo": {}}), "illegal_argument_exception", "docs array element should include an object"),
        ] {
            let (st, e) = c.call(node, Method::POST, "/m/_mget", Some(json!({"docs": [doc]}))).await;
            assert_eq!((st, e["error"]["type"].as_str(), e["error"]["reason"].as_str()), (400, Some(kind), Some(reason)), "{e}");
        }
        // URL defaults, overridden per doc.
        let (st, v) = c.call(node, Method::POST, "/m/_mget?_source=false", Some(json!({"docs": [{"_id": "1"}, {"_id": "2", "_source": true}, {"_id": "3", "_source": ["a"]}]}))).await;
        assert_eq!(st, 200, "{v}");
        let d = docs(&v);
        assert!(d[0].get("_source").is_none(), "{v}");
        assert_eq!(d[1]["_source"]["b"], json!(2), "{v}");
        assert_eq!(d[2]["_source"], json!({"a": "v3"}), "{v}");
        let (st, v) = c.call(node, Method::POST, "/m/_mget?_source=false", Some(json!({"ids": ids}))).await;
        assert_eq!(st, 200, "{v}");
        assert!(docs(&v).iter().all(|d| d["found"] == json!(true) && d.get("_source").is_none()), "{v}");
    }
}
