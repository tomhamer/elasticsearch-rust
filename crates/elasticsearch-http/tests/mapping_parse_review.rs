//! Document parsing against the mapping (production review batch b4):
//! index-wide coercion, nested `copy_to` / `include_in_root`, dotted keys
//! under nested / disabled objects, index-time field scripts (root
//! document, dependencies between them, stored output), scripts inside
//! nested queries and stored term vectors without `_source`.

use axum::body::Body;
use axum::http::Request;
use elasticsearch_engine::node::Node;
use serde_json::{json, Value};
use std::path::Path;
use tower::ServiceExt;

async fn call(r: &axum::Router, method: &str, uri: &str, body: Option<Value>) -> (u16, Value) {
    let b = match body {
        Some(v) => Body::from(v.to_string()),
        None => Body::empty(),
    };
    let req = Request::builder().method(method).uri(uri).header("content-type", "application/json").body(b).unwrap();
    let resp = r.clone().oneshot(req).await.unwrap();
    let st = resp.status().as_u16();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (st, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

fn open(dir: &Path) -> axum::Router {
    elasticsearch_http::router(Node::open(dir).unwrap())
}

/// Drop the node serving `r` and reopen it from `dir`.
async fn reopen(r: axum::Router, dir: &Path) -> axum::Router {
    drop(r);
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    open(dir)
}

async fn create(r: &axum::Router, idx: &str, body: Value) {
    let (st, v) = call(r, "PUT", &format!("/{idx}"), Some(body)).await;
    assert_eq!(st, 200, "create {idx}: {v}");
}

async fn index(r: &axum::Router, idx: &str, id: &str, doc: Value) -> (u16, Value) {
    call(r, "PUT", &format!("/{idx}/_doc/{id}?refresh=true"), Some(doc)).await
}

async fn index_ok(r: &axum::Router, idx: &str, id: &str, doc: Value) {
    let (st, v) = index(r, idx, id, doc.clone()).await;
    assert!(st == 200 || st == 201, "index {idx}/{id} {doc}: {st} {v}");
}

async fn search(r: &axum::Router, idx: &str, body: Value) -> Value {
    let (st, v) = call(r, "POST", &format!("/{idx}/_search"), Some(body.clone())).await;
    assert_eq!(st, 200, "{body} -> {v}");
    v
}

fn total(v: &Value) -> u64 {
    v["hits"]["total"]["value"].as_u64().unwrap()
}

fn ids(v: &Value) -> Vec<String> {
    let mut out: Vec<String> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap().to_string()).collect();
    out.sort();
    out
}

async fn flush(r: &axum::Router, idx: &str) {
    let (st, v) = call(r, "POST", &format!("/{idx}/_flush"), None).await;
    assert_eq!(st, 200, "{v}");
}

const ONE_SHARD: &str = r#"{"number_of_shards": 1, "number_of_replicas": 0}"#;

fn settings(extra: Value) -> Value {
    let mut s: Value = serde_json::from_str(ONE_SHARD).unwrap();
    if let (Some(o), Some(e)) = (s.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            o.insert(k.clone(), v.clone());
        }
    }
    s
}

// ---------------------------------------------------------------------------
// 198: `index.mapping.coerce: false` is the default of every field without
// its own `coerce`.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn index_mapping_coerce_false_rejects_instead_of_truncating() {
    let dir = tempfile::tempdir().unwrap();
    let r = open(dir.path());
    for (idx, extra) in [("strict", json!({})), ("strict_synth", json!({"index.mapping.source.mode": "synthetic"}))] {
        let mut s = settings(extra);
        s["index.mapping.coerce"] = json!(false);
        create(&r, idx, json!({"settings": s, "mappings": {"properties": {
            "quantity": {"type": "long"},
            "ratio": {"type": "integer", "coerce": true}
        }}}))
        .await;
        for bad in [json!(3.7), json!("3.7"), json!("3")] {
            let (st, v) = index(&r, idx, "1", json!({ "quantity": bad.clone() })).await;
            assert_eq!(st, 400, "{idx} {bad}: {v}");
            assert_eq!(v["error"]["type"], "document_parsing_exception", "{idx} {bad}: {v}");
        }
        // Nothing was written.
        assert_eq!(total(&search(&r, idx, json!({"query": {"match_all": {}}})).await), 0, "{idx}");
        // An exact long is fine; a field's own `coerce: true` overrides the index default.
        index_ok(&r, idx, "2", json!({"quantity": 3, "ratio": "4.9"})).await;
        let (st, v) = call(&r, "GET", &format!("/{idx}/_doc/2"), None).await;
        assert_eq!(st, 200, "{v}");
        assert_eq!(v["_source"]["quantity"], json!(3), "{idx}: {v}");
        let hits = search(&r, idx, json!({"query": {"term": {"ratio": 4}}})).await;
        assert_eq!(ids(&hits), vec!["2"], "{idx}: {hits}");
    }
    // Without the setting numeric values still coerce.
    create(&r, "lenient", json!({"settings": settings(json!({})), "mappings": {"properties": {"quantity": {"type": "long"}}}})).await;
    index_ok(&r, "lenient", "1", json!({"quantity": "3.7"})).await;
    let hits = search(&r, "lenient", json!({"query": {"term": {"quantity": 3}}})).await;
    assert_eq!(ids(&hits), vec!["1"], "{hits}");
}

// ---------------------------------------------------------------------------
// 229: `script` / `script_score` / script functions inside a `nested` query
// run on that path's documents.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scripts_inside_nested_queries_match_nested_documents() {
    let dir = tempfile::tempdir().unwrap();
    let r = open(dir.path());
    create(&r, "n", json!({"settings": settings(json!({})), "mappings": {"properties": {
        "items": {"type": "nested", "properties": {"price": {"type": "long"}, "tags": {"type": "nested", "properties": {"w": {"type": "long"}}}}},
        "other": {"type": "nested", "properties": {"x": {"type": "long"}}},
        "top": {"type": "long"}
    }}}))
    .await;
    index_ok(&r, "n", "1", json!({"top": 1, "items": [{"price": 10, "tags": [{"w": 3}]}, {"price": 2}]})).await;
    index_ok(&r, "n", "2", json!({"top": 2, "items": [{"price": 1}], "other": [{"x": 50}]})).await;
    index_ok(&r, "n", "3", json!({"top": 3})).await;
    for phase in ["memtable", "flushed"] {
        if phase == "flushed" {
            flush(&r, "n").await;
        }
        let nested = |q: Value| json!({"query": {"nested": {"path": "items", "query": q, "score_mode": "max"}}});
        let hits = search(&r, "n", nested(json!({"script": {"script": "doc['items.price'].value > 5"}}))).await;
        assert_eq!(ids(&hits), vec!["1"], "{phase}: {hits}");
        // Same as the equivalent range query.
        let range = search(&r, "n", nested(json!({"range": {"items.price": {"gt": 5}}}))).await;
        assert_eq!(ids(&range), ids(&hits), "{phase}");
        let hits = search(&r, "n", nested(json!({"script": {"script": "true"}}))).await;
        assert_eq!(ids(&hits), vec!["1", "2"], "{phase}: {hits}");
        let hits = search(&r, "n", nested(json!({"script_score": {"query": {"match_all": {}}, "script": "doc['items.price'].value"}}))).await;
        assert_eq!(ids(&hits), vec!["1", "2"], "{phase}: {hits}");
        assert_eq!(hits["hits"]["hits"][0]["_id"], "1", "{phase}: {hits}");
        assert_eq!(hits["hits"]["hits"][0]["_score"].as_f64(), Some(10.0), "{phase}: {hits}");
        let hits = search(&r, "n", nested(json!({"function_score": {"query": {"match_all": {}}, "functions": [{"script_score": {"script": "doc['items.price'].value"}}], "boost_mode": "replace"}}))).await;
        assert_eq!(ids(&hits), vec!["1", "2"], "{phase}: {hits}");
        assert_eq!(hits["hits"]["hits"][0]["_score"].as_f64(), Some(10.0), "{phase}: {hits}");
        // Multilevel: a script inside the inner nested path.
        let deep = json!({"query": {"nested": {"path": "items", "query": {"nested": {"path": "items.tags", "query": {"script": {"script": "doc['items.tags.w'].value == 3"}}}}}}});
        assert_eq!(ids(&search(&r, "n", deep).await), vec!["1"], "{phase}");
        // Top level: children stay hidden.
        let hits = search(&r, "n", json!({"query": {"script": {"script": "true"}}})).await;
        assert_eq!(total(&hits), 3, "{phase}: {hits}");
        let hits = search(&r, "n", json!({"query": {"script_score": {"query": {"match_all": {}}, "script": "1"}}})).await;
        assert_eq!(total(&hits), 3, "{phase}: {hits}");
    }
}

// ---------------------------------------------------------------------------
// 197: a root index-time script reads the root document, not the first
// nested child.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn root_index_time_script_reads_the_root_document_with_nested_children() {
    let dir = tempfile::tempdir().unwrap();
    let r = open(dir.path());
    create(&r, "s", json!({"settings": settings(json!({})), "mappings": {"properties": {
        "amount": {"type": "long"},
        "items": {"type": "nested", "properties": {"price": {"type": "long"}, "tags": {"type": "nested", "properties": {"w": {"type": "long"}}}}},
        "derived": {"type": "long", "script": {"source": "if (doc['amount'].size() > 0) { emit(doc['amount'].value) } else { emit(0) }"}}
    }}}))
    .await;
    index_ok(&r, "s", "0", json!({"amount": 40})).await;
    index_ok(&r, "s", "1", json!({"amount": 41, "items": [{"price": 1}]})).await;
    index_ok(&r, "s", "2", json!({"amount": 42, "items": [{"price": 1}, {"price": 2, "tags": [{"w": 1}, {"w": 2}]}]})).await;
    let check = |phase: &'static str| {
        let r = r.clone();
        async move {
            for (id, amount) in [("0", 40), ("1", 41), ("2", 42)] {
                let hits = search(&r, "s", json!({"query": {"term": {"derived": amount}}})).await;
                assert_eq!(ids(&hits), vec![id], "{phase}: derived of {id}: {hits}");
            }
            assert_eq!(total(&search(&r, "s", json!({"query": {"term": {"derived": 0}}})).await), 0, "{phase}");
            let v = search(&r, "s", json!({"size": 3, "sort": [{"derived": "desc"}], "aggs": {"sum": {"sum": {"field": "derived"}}}})).await;
            let sorted: Vec<&str> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
            assert_eq!(sorted, vec!["2", "1", "0"], "{phase}: {v}");
            assert_eq!(v["aggregations"]["sum"]["value"].as_f64(), Some(123.0), "{phase}: {v}");
        }
    };
    check("memtable").await;
    flush(&r, "s").await;
    check("flushed").await;
}

// ---------------------------------------------------------------------------
// 244: index-time scripts read other scripted fields (run first, once).

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn index_time_scripts_read_other_scripted_fields() {
    let dir = tempfile::tempdir().unwrap();
    let r = open(dir.path());
    create(&r, "chain", json!({"settings": settings(json!({})), "mappings": {"properties": {
        "amount": {"type": "long"},
        // Reverse-alphabetical chain: a3 <- b2 <- c1 <- _source.
        "c1": {"type": "long", "script": {"source": "if (params._source.amount != null) { emit(params._source.amount * 2) }"}},
        "b2": {"type": "long", "script": {"source": "if (doc['c1'].size() > 0) { emit(doc['c1'].value + 1) } else { emit(-1) }"}},
        "a3": {"type": "long", "script": {"source": "emit(doc['b2'].value + 1)"}},
        "base_total": {"type": "long", "script": {"source": "emit(10)"}},
        "adjusted_total": {"type": "long", "script": {"source": "if (doc['base_total'].size() > 0) { emit(doc['base_total'].value) } else { emit(0) }"}}
    }}}))
    .await;
    index_ok(&r, "chain", "1", json!({"amount": 5})).await;
    // `amount` genuinely missing: the fallback applies.
    index_ok(&r, "chain", "2", json!({"other": 1})).await;
    flush(&r, "chain").await;
    for (field, value, want) in [
        ("c1", 10, vec!["1"]),
        ("b2", 11, vec!["1"]),
        ("a3", 12, vec!["1"]),
        ("b2", -1, vec!["2"]),
        ("a3", 0, vec!["2"]),
        ("base_total", 10, vec!["1", "2"]),
        ("adjusted_total", 10, vec!["1", "2"]),
    ] {
        let hits = search(&r, "chain", json!({"query": {"term": {field: value}}})).await;
        assert_eq!(ids(&hits), want, "{field}={value}: {hits}");
    }
    assert_eq!(total(&search(&r, "chain", json!({"query": {"term": {"adjusted_total": 0}}})).await), 0);

    // A loop rejects the document; nothing is written.
    create(&r, "loop", json!({"settings": settings(json!({})), "mappings": {"properties": {
        "field1": {"type": "long", "script": {"source": "if (doc['field2'].size() > 0) { emit(doc['field2'].value + 2) } else { emit(0) }"}},
        "field2": {"type": "long", "script": {"source": "if (doc['field1'].size() > 0) { emit(doc['field1'].value + 2) } else { emit(0) }"}}
    }}}))
    .await;
    let (st, v) = index(&r, "loop", "1", json!({})).await;
    assert_eq!(st, 400, "{v}");
    assert_eq!(v["error"]["type"], "document_parsing_exception", "{v}");
    assert!(v["error"]["reason"].as_str().unwrap().contains("Error executing script on field [field1]"), "{v}");
    assert!(v.to_string().contains("Loop in field resolution detected: field2->field1->field2"), "{v}");
    assert_eq!(total(&search(&r, "loop", json!({})).await), 0);
}

// ---------------------------------------------------------------------------
// 245: a `store: true` scripted field stores the values its script emitted.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stored_scripted_fields_keep_the_emitted_values() {
    let dir = tempfile::tempdir().unwrap();
    let mut r = open(dir.path());
    for (idx, source) in [("st", json!({})), ("st_nosrc", json!({"enabled": false}))] {
        create(&r, idx, json!({"settings": settings(json!({})), "mappings": {"_source": source, "properties": {
            "amount": {"type": "long"},
            "computed": {"type": "long", "store": true, "doc_values": false, "script": {"source": "emit(params._source.amount * 2)"}},
            "thrice": {"type": "long", "store": true, "index": false, "script": {"source": "emit(params._source.amount * 3)"}}
        }}}))
        .await;
        index_ok(&r, idx, "1", json!({"amount": 7})).await;
    }
    for phase in ["memtable", "flushed", "reopened"] {
        if phase == "flushed" {
            flush(&r, "st").await;
            flush(&r, "st_nosrc").await;
        }
        if phase == "reopened" {
            r = reopen(r, dir.path()).await;
        }
        for idx in ["st", "st_nosrc"] {
            let (st, v) = call(&r, "GET", &format!("/{idx}/_doc/1?stored_fields=computed,thrice"), None).await;
            assert_eq!(st, 200, "{phase} {idx}: {v}");
            assert_eq!(v["fields"]["computed"], json!([14]), "{phase} {idx}: {v}");
            assert_eq!(v["fields"]["thrice"], json!([21]), "{phase} {idx}: {v}");
            let hits = search(&r, idx, json!({"stored_fields": ["computed", "thrice"], "query": {"term": {"computed": 14}}})).await;
            assert_eq!(hits["hits"]["hits"][0]["fields"]["computed"], json!([14]), "{phase} {idx}: {hits}");
            assert_eq!(hits["hits"]["hits"][0]["fields"]["thrice"], json!([21]), "{phase} {idx}: {hits}");
        }
    }
}

// ---------------------------------------------------------------------------
// 196: `copy_to` from a nested object reaches the root / ancestor target;
// `include_in_root` of an inner nested object reaches the root.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nested_copy_to_and_include_in_root_reach_their_documents() {
    let dir = tempfile::tempdir().unwrap();
    let mut r = open(dir.path());
    create(&r, "cp", json!({"settings": settings(json!({})), "mappings": {"properties": {
        "catch_all": {"type": "text"},
        "items": {"type": "nested", "properties": {
            "name": {"type": "keyword", "copy_to": ["catch_all", "items.all"]},
            "all": {"type": "text"},
            "parts": {"type": "nested", "properties": {
                "pname": {"type": "keyword", "copy_to": ["items.all", "catch_all"]}
            }}
        }}
    }}}))
    .await;
    index_ok(&r, "cp", "1", json!({"items": [{"name": "invoice", "parts": [{"pname": "bolt"}, {"pname": "nut"}]}, {"name": "receipt"}]})).await;
    index_ok(&r, "cp", "2", json!({"items": [{"name": "other"}]})).await;
    create(&r, "inc", json!({"settings": settings(json!({})), "mappings": {"properties": {
        "outer": {"type": "nested", "properties": {
            "k": {"type": "keyword"},
            "inner": {"type": "nested", "include_in_root": true, "properties": {"v": {"type": "keyword"}}},
            "both": {"type": "nested", "include_in_parent": true, "include_in_root": true, "properties": {"b": {"type": "keyword"}}}
        }}
    }}}))
    .await;
    // `include_in_root` of a nested object inside one not included in the
    // root is kept (ES drops it only when the parent already reaches the root).
    let (st, m) = call(&r, "GET", "/inc/_mapping", None).await;
    assert_eq!(st, 200, "{m}");
    assert_eq!(m["inc"]["mappings"]["properties"]["outer"]["properties"]["both"]["include_in_root"], json!(true), "{m}");
    index_ok(&r, "inc", "1", json!({"outer": [{"k": "a", "inner": [{"v": "x"}, {"v": "y"}], "both": [{"b": "z"}]}]})).await;
    index_ok(&r, "inc", "2", json!({"outer": [{"k": "b", "inner": [{"v": "w"}]}]})).await;
    for phase in ["memtable", "flushed", "reopened"] {
        if phase == "flushed" {
            flush(&r, "cp").await;
            flush(&r, "inc").await;
        }
        if phase == "reopened" {
            r = reopen(r, dir.path()).await;
        }
        // Root target.
        for (q, want) in [("invoice", vec!["1"]), ("receipt", vec!["1"]), ("bolt", vec!["1"]), ("other", vec!["2"])] {
            let hits = search(&r, "cp", json!({"query": {"match": {"catch_all": q}}})).await;
            assert_eq!(ids(&hits), want, "{phase} catch_all:{q}: {hits}");
        }
        // Not on the nested children.
        let hits = search(&r, "cp", json!({"query": {"nested": {"path": "items", "query": {"match": {"catch_all": "invoice"}}}}})).await;
        assert_eq!(total(&hits), 0, "{phase}: {hits}");
        // Ancestor (nested) target: `items.all` of the item owning the part.
        let hits = search(&r, "cp", json!({"query": {"nested": {"path": "items", "query": {"bool": {"must": [{"match": {"items.all": "bolt"}}, {"match": {"items.all": "invoice"}}]}}}}})).await;
        assert_eq!(ids(&hits), vec!["1"], "{phase}: {hits}");
        let hits = search(&r, "cp", json!({"query": {"nested": {"path": "items", "query": {"bool": {"must": [{"match": {"items.all": "bolt"}}, {"match": {"items.all": "receipt"}}]}}}}})).await;
        assert_eq!(total(&hits), 0, "{phase}: a part's value reaches only its own item: {hits}");
        // Same-scope fields stay nested only.
        assert_eq!(total(&search(&r, "cp", json!({"query": {"term": {"items.name": "invoice"}}})).await), 0, "{phase}");
        let hits = search(&r, "cp", json!({"query": {"nested": {"path": "items", "query": {"term": {"items.name": "invoice"}}}}})).await;
        assert_eq!(ids(&hits), vec!["1"], "{phase}");
        // Multilevel include flags.
        for (field, value, want) in [("outer.inner.v", "x", vec!["1"]), ("outer.inner.v", "w", vec!["2"]), ("outer.both.b", "z", vec!["1"])] {
            let hits = search(&r, "inc", json!({"query": {"term": {field: value}}})).await;
            assert_eq!(ids(&hits), want, "{phase} root {field}:{value}: {hits}");
        }
        // `include_in_parent`: on the `outer` document too; `inner` is not.
        let hits = search(&r, "inc", json!({"query": {"nested": {"path": "outer", "query": {"term": {"outer.both.b": "z"}}}}})).await;
        assert_eq!(ids(&hits), vec!["1"], "{phase}: {hits}");
        let hits = search(&r, "inc", json!({"query": {"nested": {"path": "outer", "query": {"term": {"outer.inner.v": "x"}}}}})).await;
        assert_eq!(total(&hits), 0, "{phase}: {hits}");
        // `outer` itself is not included anywhere.
        assert_eq!(total(&search(&r, "inc", json!({"query": {"term": {"outer.k": "a"}}})).await), 0, "{phase}");
        // The source is unchanged.
        let (_, d) = call(&r, "GET", "/cp/_doc/1", None).await;
        assert_eq!(d["_source"], json!({"items": [{"name": "invoice", "parts": [{"pname": "bolt"}, {"pname": "nut"}]}, {"name": "receipt"}]}), "{phase}");
    }
}

// ---------------------------------------------------------------------------
// 248: a dotted source key below a nested / disabled object enters that
// object's scope like the object-shaped input.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dotted_keys_enter_nested_and_disabled_object_scopes() {
    let dir = tempfile::tempdir().unwrap();
    let mut r = open(dir.path());
    create(&r, "dk", json!({"settings": settings(json!({})), "mappings": {"properties": {
        "items": {"type": "nested", "properties": {
            "name": {"type": "keyword"},
            "tags": {"type": "nested", "properties": {"w": {"type": "keyword"}}}
        }},
        "wrap": {"properties": {"items": {"type": "nested", "properties": {"name": {"type": "keyword"}}}}}
    }}}))
    .await;
    index_ok(&r, "dk", "dotted", json!({"items.name": "invoice", "items.tags.w": "deep", "wrap.items.name": "wrapped"})).await;
    index_ok(&r, "dk", "object", json!({"items": {"name": "invoice", "tags": {"w": "deep"}}, "wrap": {"items": {"name": "wrapped"}}})).await;
    // A predeclared child of a disabled object, given as a dotted key.
    let (st, v) = call(&r, "PUT", "/dis", Some(json!({"settings": settings(json!({})), "mappings": {"properties": {
        "meta": {"type": "object", "enabled": false, "properties": {"x": {"type": "keyword"}}}
    }}}))).await;
    let disabled = st == 200;
    if disabled {
        index_ok(&r, "dis", "dotted", json!({"meta.x": "secret"})).await;
        index_ok(&r, "dis", "object", json!({"meta": {"x": "secret"}})).await;
    } else {
        eprintln!("a mapped child of a disabled object is rejected here: {v}");
    }
    for phase in ["memtable", "flushed", "reopened"] {
        if phase == "flushed" {
            flush(&r, "dk").await;
        }
        if phase == "reopened" {
            r = reopen(r, dir.path()).await;
        }
        let nested = |path: &str, field: &str, value: &str| json!({"query": {"nested": {"path": path, "query": {"term": {field: value}}}}});
        assert_eq!(ids(&search(&r, "dk", nested("items", "items.name", "invoice")).await), vec!["dotted", "object"], "{phase}");
        assert_eq!(ids(&search(&r, "dk", nested("items.tags", "items.tags.w", "deep")).await), vec!["dotted", "object"], "{phase}");
        assert_eq!(ids(&search(&r, "dk", nested("wrap.items", "wrap.items.name", "wrapped")).await), vec!["dotted", "object"], "{phase}");
        // Not on the root document.
        for (field, value) in [("items.name", "invoice"), ("items.tags.w", "deep"), ("wrap.items.name", "wrapped")] {
            assert_eq!(total(&search(&r, "dk", json!({"query": {"term": {field: value}}})).await), 0, "{phase} root {field}");
        }
        if disabled {
            assert_eq!(total(&search(&r, "dis", json!({"query": {"term": {"meta.x": "secret"}}})).await), 0, "{phase}");
            assert_eq!(total(&search(&r, "dis", json!({"query": {"match_all": {}}})).await), 2, "{phase}");
        }
    }
}

// ---------------------------------------------------------------------------
// 233: stored term vectors survive `_source` excludes / `_source` disabled.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stored_term_vectors_survive_pruned_or_disabled_source() {
    let dir = tempfile::tempdir().unwrap();
    let mut r = open(dir.path());
    let props = json!({
        "body": {"type": "text", "term_vector": "with_positions_offsets", "fields": {"raw": {"type": "keyword"}}},
        "title": {"type": "text", "term_vector": "with_positions"},
        "plain": {"type": "text"}
    });
    for (idx, source) in [("tv_full", json!({})), ("tv_ex", json!({"excludes": ["body", "title"]})), ("tv_dis", json!({"enabled": false}))] {
        create(&r, idx, json!({"settings": settings(json!({})), "mappings": {"_source": source, "properties": props.clone()}})).await;
        index_ok(&r, idx, "1", json!({"body": "the quick brown fox quick", "title": ["one two", "two"], "plain": "x"})).await;
    }
    let tv = |r: axum::Router, idx: &'static str| async move {
        let (st, v) = call(&r, "GET", &format!("/{idx}/_termvectors/1"), None).await;
        assert_eq!(st, 200, "{idx}: {v}");
        assert_eq!(v["found"], json!(true), "{idx}: {v}");
        v["term_vectors"].clone()
    };
    for phase in ["refreshed", "flushed", "merged", "reopened"] {
        match phase {
            "flushed" => {
                for idx in ["tv_full", "tv_ex", "tv_dis"] {
                    flush(&r, idx).await;
                }
            }
            "merged" => {
                for idx in ["tv_full", "tv_ex", "tv_dis"] {
                    let (st, v) = call(&r, "POST", &format!("/{idx}/_forcemerge?max_num_segments=1"), None).await;
                    assert_eq!(st, 200, "{v}");
                }
            }
            "reopened" => r = reopen(r, dir.path()).await,
            _ => {}
        }
        let full = tv(r.clone(), "tv_full").await;
        let quick = &full["body"]["terms"]["quick"];
        assert_eq!(quick["term_freq"], json!(2), "{phase}: {full}");
        assert_eq!(quick["tokens"][1]["position"], json!(4), "{phase}: {full}");
        assert_eq!(quick["tokens"][1]["start_offset"], json!(20), "{phase}: {full}");
        assert_eq!(full["title"]["terms"]["two"]["term_freq"], json!(2), "{phase}: {full}");
        for idx in ["tv_ex", "tv_dis"] {
            assert_eq!(tv(r.clone(), idx).await, full, "{phase} {idx}");
        }
    }
    // Excluded fields stay out of `_source` and `stored_fields`.
    let (_, d) = call(&r, "GET", "/tv_ex/_doc/1?stored_fields=*", None).await;
    assert!(d.get("fields").is_none(), "{d}");
    let (_, d) = call(&r, "GET", "/tv_ex/_doc/1", None).await;
    assert_eq!(d["_source"], json!({"plain": "x"}), "{d}");
}

/// Production review #245 follow-up: ES rejects an index-time script on a
/// field that is neither indexed nor has doc values (its values could go
/// nowhere).
#[tokio::test]
async fn scripted_field_needs_index_or_doc_values() {
    let tmp = tempfile::tempdir().unwrap();
    let r = open(tmp.path());
    let (st, v) = call(&r, "PUT", "/nowhere", Some(json!({"mappings": {"properties": {"x": {"type": "long", "index": false, "doc_values": false, "script": {"source": "emit(1)"}}}}}))).await;
    assert_eq!((st, v["error"]["type"].as_str()), (400, Some("mapper_parsing_exception")), "{v}");
    assert!(v.to_string().contains("Cannot define script on field with index:false and doc_values:false"), "{v}");
    let (st, v) = call(&r, "PUT", "/somewhere", Some(json!({"mappings": {"properties": {"x": {"type": "long", "index": false, "script": {"source": "emit(1)"}}}}}))).await;
    assert_eq!(st, 200, "doc values alone are enough: {v}");
}
